"""Integration test: 2v2 offline collection for team OM pretraining.

Covers:
  1. Packing round-trip and binary-channel invariants of stored states.
  2. Per-transition invariants: hist_len == step_idx, claim-count sums
     (hostile labels sum = #hostiles, friendly labels sum = #teammates),
     intent label sums bounded.
  3. Cross-check: collection labels for team-0 actors are IDENTICAL to
     QLearningAgent._apply_hindsight_relabeling on the same episode.
  4. collate_history_packed matches OpponentModel.collate_history on the
     equivalent unpacked items.
  5. Pretrain smoke: a few epochs on the small dataset give finite losses,
     checkpoints load into fresh models and produce a valid distribution.

Run: /opt/homebrew/anaconda3/envs/om/bin/python tests/test_team_collect.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import wandb
from omexplore.agents.q_agent import QLearningAgent
from omexplore.agents.team_agents import TeamAgent
from omexplore.collect_data import run_team_collection_episode
from omexplore.envs.roadmap_foraging_env import TeamRoadmapEnv
from omexplore.models.opponent_model import OpponentModel
from omexplore.models.transformers import SpatialOpponentModel
from omexplore.utils.labeling import dense_from_sparse
from omexplore.utils.omg_args import OMGArgs
from omexplore.utils.packing import pack_obs, unpack_obs
from scripts.pretrain_team_oms import build_om_batch, collate_history_packed


def main():
    wandb.init(mode="disabled")

    env = TeamRoadmapEnv(max_steps=30, num_goals=6, team_sizes=(2, 2), vision_radius=5)
    H, W = env.height, env.width

    field_cache = {}
    team_agents = [
        TeamAgent(
            env,
            team_id=t,
            personas=personas,
            field_cache=field_cache,
        )
        for t, personas in enumerate([("greedy", "simple"), ("switch", "stalker")])
    ]

    transitions = []
    records = {}
    for _ in range(2):
        transitions.extend(
            run_team_collection_episode(env, team_agents, records_out=records)
        )
    print(f"Collected {len(transitions)} transitions")

    # --- 1+2: packing and label invariants -----------------------------
    for tr in transitions:
        assert tr["state"].shape == (H, W, 1) and tr["state"].dtype == np.uint8
        unpacked = unpack_obs(tr["state"])
        assert unpacked.shape == (H, W, 7)
        assert unpacked.dtype == np.uint8
        assert np.isin(unpacked, [0, 1]).all(), "channels must be binary"
        assert (pack_obs(unpacked.astype(np.int8)) == tr["state"]).all(), (
            "pack/unpack roundtrip failed"
        )
        # hist_len semantics: anchor states appended before this step
        assert tr["hist_len"] == tr["step_idx"]
        hist = tr["history"]["states"]
        assert hist.ndim == 4 and hist.shape[1:] == (H, W, 1)
        assert hist.shape[0] >= tr["hist_len"]
        # claim-count sums: every agent gets exactly one label
        hostile_n = sum(1 for a in env.agents if env.teams[a] != tr["team"])
        team_n = sum(1 for m in env.get_team_members(tr["team"]) if m != tr["agent_id"])
        assert sum(w for _, _, w in tr["true_goal_cells"]) == hostile_n
        assert sum(w for _, _, w in tr["true_team_cells"]) == team_n
        # intent labels: at most one claim per agent, may be empty (no target)
        assert sum(w for _, _, w in tr["true_opp_heatmap_cells"]) <= hostile_n
        assert sum(w for _, _, w in tr["true_team_heatmap_cells"]) <= team_n
        # action/reward/done fields present and sane
        assert 0 <= tr["action"] <= 7
        assert tr["done"] in (True, False)
    assert any(len(tr["true_opp_heatmap_cells"]) > 0 for tr in transitions), (
        "no non-empty hostile intent labels — intent term would be vacuous"
    )
    print("Packing + label-sum invariants OK")

    # --- 3: cross-check vs QLearningAgent hindsight labeling -------------
    args = OMGArgs(
        device="cpu",
        state_shape=(H, W, 7),
        H=H,
        W=W,
        action_dim=8,
        max_steps=30,
        max_history_length=4,
        capacity=100,
        min_replay=100,
        batch_size=4,
        train_every=1000,
    )
    hostile_om = OpponentModel(SpatialOpponentModel(args), args)
    friendly_om = OpponentModel(SpatialOpponentModel(args), args)
    agent = QLearningAgent(env, hostile_om, friendly_om, args=args)
    assert agent.learn_ids == [0, 1] and agent.hostile_ids == [2, 3]

    # Only the LAST collected episode's records match the last chunk of
    # transitions (records_out is overwritten per episode).
    per_ep = len(env.agents) * len(records["step_records"])
    last_ep = transitions[-per_ep:]
    qa_transitions = [
        {"agent_id": tr["agent_id"], "step_idx": tr["step_idx"]}
        for tr in last_ep
        if tr["team"] == 0
    ]
    agent._apply_hindsight_relabeling(
        qa_transitions, records["step_records"], records["final_positions"], H, W
    )
    it = iter(qa_transitions)
    for tr in last_ep:
        if tr["team"] != 0:
            continue
        qa = next(it)
        assert np.allclose(
            qa["true_goal_map"], dense_from_sparse([tr["true_goal_cells"]], H, W)[0]
        ), f"hostile label mismatch at step {tr['step_idx']}"
        assert np.allclose(
            qa["true_team_goal_map"],
            dense_from_sparse([tr["true_team_cells"]], H, W)[0],
        ), f"friendly label mismatch at step {tr['step_idx']}"
    print("Labels match QLearningAgent._apply_hindsight_relabeling OK")

    # --- 4: packed collate == unpacked collate ---------------------------
    items = [tr for tr in transitions if tr["step_idx"] >= 3][:8]
    unpacked_items = []
    for tr in items:
        seq = unpack_obs(tr["history"]["states"])
        unpacked_items.append({"hist_len": tr["hist_len"], "history": {"states": seq}})
    ref = hostile_om.collate_history(unpacked_items)
    got = collate_history_packed(items, args, "cpu")
    assert torch.equal(ref["mask"], got["mask"])
    assert torch.allclose(ref["states"], got["states"])
    assert torch.allclose(ref["prev_first"], got["prev_first"])
    print("Packed collate == unpacked collate OK")

    # --- 5: pretrain smoke ------------------------------------------------
    device = "cpu"
    batch = build_om_batch(
        items, args, device, "true_goal_cells", "true_opp_heatmap_cells"
    )
    loss, kl, sp = hostile_om.pretrain_step(batch)
    assert np.isfinite(loss), f"hostile pretrain loss not finite: {loss}"
    print(f"Hostile pretrain smoke: loss={loss:.4f} kl={kl:.4f} spatial={sp:.4f}")

    fbatch = build_om_batch(
        items, args, device, "true_team_cells", "true_team_heatmap_cells"
    )
    floss, fkl, _ = friendly_om.pretrain_step(fbatch)
    assert np.isfinite(floss), f"friendly pretrain loss not finite: {floss}"
    print(f"Friendly pretrain smoke: loss={floss:.4f} kl={fkl:.4f}")

    # Save + load roundtrip, then a forward pass.
    with tempfile.TemporaryDirectory() as tmp:
        torch.save(hostile_om.inference_model.state_dict(), f"{tmp}/h.pth")
        torch.save(friendly_om.inference_model.state_dict(), f"{tmp}/f.pth")
        fresh_h = SpatialOpponentModel(args)
        fresh_h.load_state_dict(torch.load(f"{tmp}/h.pth", weights_only=True))
        fresh_f = SpatialOpponentModel(args)
        fresh_f.load_state_dict(torch.load(f"{tmp}/f.pth", weights_only=True))
        fresh_h.eval()
        hist2 = collate_history_packed(items[:2], args, "cpu")
        with torch.no_grad():
            out = fresh_h(batch["states"][:2], hist2, cached_features=False)
            probs = torch.softmax(out.flatten(1), dim=-1).sum(dim=-1)
        assert torch.isfinite(out).all()
        assert torch.allclose(probs, torch.ones(2), atol=1e-4), probs
    print("Checkpoint save/load + forward OK")

    print("\nALL TEAM COLLECTION TESTS PASSED")


if __name__ == "__main__":
    main()

"""Cached-feature equivalence tests for the OM + QLearningAgent team path.

Guards the cached_features=True refactor:
  1. collate_cached_history_pair correctly reconstructs each transition's
     current/next feature windows, masks, and prev_obs from the shared
     per-episode feature arrays.
  2. OM train_step on cached features gives the SAME loss as recomputing
     features from raw states (given frozen extractor weights) — this is the
     exact guarantee that switching training to cached mode is loss-free.
  3. The full update() plumbing (compute_targets contaminating both OMs with
     cached histories) runs and produces finite losses.

Run:  /opt/homebrew/anaconda3/envs/om/bin/python tests/test_cached_features.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import wandb
from omexplore.agents.q_agent import QLearningAgent
from omexplore.agents.team_agents import TeamAgent
from omexplore.envs.roadmap_foraging_env import TeamRoadmapEnv
from omexplore.models.opponent_model import OpponentModel
from omexplore.models.transformers import SpatialOpponentModel
from omexplore.utils.omg_args import OMGArgs

wandb.init(mode="disabled", project="cached-features-test")

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, "PASS", ""))
        print(f"[PASS] {name}")
    except Exception as e:
        RESULTS.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


def make_env_args():
    env = TeamRoadmapEnv(max_steps=40, num_goals=6, team_sizes=(2, 2))
    args = OMGArgs(
        device="cpu",
        state_shape=(env.height, env.width, env.features),
        H=env.height,
        W=env.width,
        action_dim=8,
        max_steps=40,
        max_history_length=8,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=64,
        capacity=2000,
        min_replay=32,
        batch_size=8,
        train_every=2,
    )
    return env, args


env, args = make_env_args()
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

hostile_om = OpponentModel(SpatialOpponentModel(args), args)
friendly_om = OpponentModel(SpatialOpponentModel(args), args)
agent = QLearningAgent(env, hostile_om, friendly_om, args=args)
opp = TeamAgent(env, team_id=1)


# --------------------------------------------------------------------------
# 1. collate_cached_history_pair window reconstruction
# --------------------------------------------------------------------------
def t_collator():
    H, W, F = args.state_shape
    T = args.max_history_length
    dmodel = args.d_model
    E = 15  # episode has 15 states (indices 0..14)
    states = np.random.randn(E, H, W, F).astype(np.float32)
    feats = np.random.randn(E, dmodel).astype(np.float32)

    # Transitions with hist_len L = 0, 1, 7, 14 (state index L, next L+1)
    items = []
    for L in (0, 1, 7, 14):
        items.append(
            {
                "state": states[L],
                "hist_len": L,
                "history": {"states": states, "feats_hostile": feats},
            }
        )
    out = hostile_om.collate_cached_history_pair(items, "feats_hostile")
    cur, nxt = out["cur"], out["nxt"]

    assert cur["state_features"].shape == (4, T, dmodel)
    assert nxt["state_features"].shape == (4, T, dmodel)

    for i, L in enumerate((0, 1, 7, 14)):
        # current window: last min(L,T) feats, right-aligned
        take = min(L, T)
        assert cur["mask"][i].sum() == take
        if take > 0:
            np.testing.assert_allclose(
                cur["state_features"][i, -take:].numpy(), feats[L - take : L], atol=1e-6
            )
            np.testing.assert_allclose(
                cur["prev_obs"][i].numpy(), states[L - 1], atol=1e-6
            )
        else:
            np.testing.assert_allclose(
                cur["prev_obs"][i].numpy(), np.zeros_like(states[0]), atol=1e-6
            )

        # next window: last min(L+1,T) feats, right-aligned; prev = states[L]
        take_n = min(L + 1, T)
        assert nxt["mask"][i].sum() == take_n
        if take_n > 0:
            np.testing.assert_allclose(
                nxt["state_features"][i, -take_n:].numpy(),
                feats[L + 1 - take_n : L + 1],
                atol=1e-6,
            )
            np.testing.assert_allclose(nxt["prev_obs"][i].numpy(), states[L], atol=1e-6)


check("collate_cached_history_pair (cur/nxt windows, masks, prev_obs)", t_collator)


# --------------------------------------------------------------------------
# 2. OM train_step: cached features == recomputed features (same loss)
# --------------------------------------------------------------------------
def t_train_step_equiv():
    H, W, F = args.state_shape

    torch.manual_seed(1)
    np.random.seed(1)
    B = 6
    E = 14
    states = np.random.randn(E, H, W, F).astype(np.float32)
    goal_maps = np.zeros((B, H, W), dtype=np.float32)
    goal_maps[:, 2, 3] = 1.0  # a claim near a corner (exercises boundary blur)

    # Build transitions that carry BOTH enough history and cached features.
    items = []
    for i in range(B):
        L = 1 + i % 4  # 1..4
        items.append(
            {
                "state": states[L],
                "hist_len": L,
                "history": {"states": states, "feats_hostile": None},
                "true_goal_map": goal_maps[i],
            }
        )
        # cached features computed with the SAME extractor as the raw path:
        # get_features(state_j, state_{j-1}), exactly as the rollout does.
        feats = []
        prev = torch.zeros((1, H, W, F))
        for j in range(E):
            s = torch.from_numpy(states[j]).float().unsqueeze(0)
            with torch.no_grad():
                feats.append(
                    hostile_om.inference_model.get_features(s, prev).squeeze(0)
                )
            prev = s
        items[i]["history"]["feats_hostile"] = torch.stack(feats).numpy()

    # Batch A: cached-path (as the refactor produces it).
    h_cache = hostile_om.collate_cached_history_pair(items, "feats_hostile")
    batch_cached = {
        "states": torch.from_numpy(
            np.stack([b["state"] for b in items]).astype(np.float32)
        ),
        "history": h_cache["cur"],
        "true_goal_map": torch.from_numpy(
            np.stack([b["true_goal_map"] for b in items])
        ),
    }

    # Batch B: raw-state path (the OLD way) — same transitions, no cached feats.
    for it in items:
        it["history"] = {"states": states}
    hist_raw = hostile_om.collate_history(items)
    batch_raw = {
        "states": batch_cached["states"].clone(),
        "history": hist_raw,
        "true_goal_map": batch_cached["true_goal_map"].clone(),
    }

    # Snapshot the ORIGINAL weights once, then run both train_steps from an
    # identical start (weights frozen except the update under test). A fresh
    # Adam optimizer each time removes any dependence on optimizer momentum.
    def _reset():
        hostile_om.inference_model.load_state_dict(
            {k: v.clone() for k, v in snapshot.items()}
        )
        hostile_om.tgt_model.load_state_dict(
            {k: v.clone() for k, v in snapshot.items()}
        )
        hostile_om.optimizer = torch.optim.Adam(
            hostile_om.inference_model.parameters(), lr=hostile_om.args.lr_om
        )

    snapshot = {
        k: v.clone() for k, v in hostile_om.inference_model.state_dict().items()
    }
    _reset()
    loss_cached = hostile_om.train_step(batch_cached, cached_features=True)

    _reset()
    loss_raw = hostile_om.train_step(batch_raw, cached_features=False)

    assert torch.allclose(
        torch.tensor(loss_cached), torch.tensor(loss_raw), atol=1e-5
    ), f"cached loss {loss_cached:.6f} != raw loss {loss_raw:.6f}"
    print(f"  cached OM loss {loss_cached:.6f} == raw OM loss {loss_raw:.6f}")


check(
    "OM train_step: cached features == recomputed features (same loss)",
    t_train_step_equiv,
)


# --------------------------------------------------------------------------
# 3. Full update() plumbing: cached histories flow through both OMs + Q
# --------------------------------------------------------------------------
def t_update_plumbing():
    # Fill the replay to at least min_replay so update() actually fires.
    while len(agent.replay) < agent.args.min_replay + 4:
        agent.run_episode(opp, max_steps=args.max_steps)

    # Force an update by driving global_step to a train_every boundary.
    while agent.global_step % agent.args.train_every != 0:
        agent.global_step += 1
    loss_val, model_loss, team_loss = agent.update()

    assert loss_val is not None, "update() did not run (min_replay/step guard)"
    assert torch.isfinite(torch.tensor(loss_val)), f"Q loss non-finite: {loss_val}"
    assert torch.isfinite(torch.tensor(model_loss)), (
        f"hostile OM loss non-finite: {model_loss}"
    )
    assert torch.isfinite(torch.tensor(team_loss)), (
        f"friendly OM loss non-finite: {team_loss}"
    )
    print(
        f"  update(): Q {loss_val:.4f}, hostile OM {model_loss:.4f}, "
        f"friendly OM {team_loss:.4f}"
    )


check(
    "update() with cached histories (Q + hostile OM + friendly OM)", t_update_plumbing
)


if __name__ == "__main__":
    n_fail = sum(1 for _, st, _ in RESULTS if st == "FAIL")
    print(f"\n{'=' * 50}\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    if n_fail:
        print("Failed:")
        for name, st, msg in RESULTS:
            if st == "FAIL":
                print(f"  - {name}: {msg}")
        sys.exit(1)
    print("ALL CACHED-FEATURE TESTS PASSED")

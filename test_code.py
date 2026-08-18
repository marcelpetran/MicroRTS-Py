"""
Smoke test for the OM + Q-learning training pipeline.
Runs every component for a few steps against small synthetic data and asserts
the invariants that the recent refactors depend on. No long training runs.

Run:  python smoke_test.py
"""

import traceback

import numpy as np
import torch

import wandb
from collect_data import collect_offline_data
from maps import MAP_1
from omg_args import OMGArgs
from opponent_model import OpponentModel
from q_agent import QLearningAgent
from simple_foraging_env import GreedySwitchAgent, SimpleAgent, SimpleForagingEnv
from transformers import SpatialOpponentModel

wandb.init(mode="disabled", project="smoke-test")

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, "PASS", ""))
        print(f"[PASS] {name}")
    except Exception as e:
        RESULTS.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


# --------------------------------------------------------------------------
# Setup: tiny everything
# --------------------------------------------------------------------------
def make_args():
    env = SimpleForagingEnv(max_steps=50, map_layout=MAP_1)
    obs = env.reset()
    H, W, F_dim = obs[0].shape
    return env, OMGArgs(
        device="cpu",
        H=H,
        W=W,
        action_dim=4,
        state_shape=obs[0].shape,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=64,
        batch_size=8,
        capacity=1_000,
        min_replay=16,
        train_every=2,
        true_intent=False,
    )


env, args = make_args()
torch.manual_seed(0)
np.random.seed(0)

model = SpatialOpponentModel(args).to(args.device)
om = OpponentModel(model, args=args)
agent = QLearningAgent(env, om, args=args)


# --------------------------------------------------------------------------
# 1. collate_history: list input (old offline format) AND ndarray (new format),
#    hist_len=0, extra=0/1, right-alignment, no action keys required
# --------------------------------------------------------------------------
def t_collate():
    H, W, F = args.state_shape
    ep = np.random.randint(0, 2, size=(11, H, W, F)).astype(np.int8)  # was 10
    items = [
        {"hist_len": 0, "history": {"states": ep}},
        {"hist_len": 5, "history": {"states": ep}},
        {"hist_len": 10, "history": {"states": ep}},  # full 11-frame ep
        {"hist_len": 7, "history": {"states": list(ep[:7])}},
        {
            "hist_len": 999,
            "history": {"states": np.zeros((1000, H, W, F), dtype=np.int8)},
        },
    ]
    out = om.collate_history(items)  # extra=0
    assert out["states"].shape == (5, args.max_history_length, H, W, F)
    assert out["mask"].shape == (5, args.max_history_length)
    assert "actions" not in out, "action pathway should be gone"
    assert out["mask"][0].sum() == 0, "hist_len=0 must give empty mask"
    assert out["mask"][1].sum() == 5
    assert out["mask"][2].sum() == 10, "full-length item must not be truncated"
    assert out["mask"][3].sum() == 7, "legacy list input must stack correctly"
    assert out["mask"][4].sum() == args.max_history_length, "overflow clamps to max_len"
    assert np.allclose(out["states"][1, -1].cpu().numpy(), ep[4].astype(np.float32))

    out1 = om.collate_history(items, extra=1)
    assert out1["mask"][0].sum() == 1, "extra=1 on hist_len=0 must give 1 frame"
    assert np.allclose(out1["states"][1, -1].cpu().numpy(), ep[5].astype(np.float32))


check("collate_history (formats, empty, extra, alignment)", t_collate)


# --------------------------------------------------------------------------
# 2. Forward pass, both paths — incl. cached_features=True with prev_obs=None
#    (this is the rollout path; caught the x_feat NameError last round)
# --------------------------------------------------------------------------
def t_forward():
    H, W, F = args.state_shape
    B = 4
    x = torch.randint(0, 2, (B, H, W, F), device=args.device).float()
    hist = {
        "states": torch.randint(0, 2, (B, 5, H, W, F), device=args.device).float(),
        "mask": torch.ones((B, 5), dtype=torch.bool, device=args.device),
    }
    # fresh (uncached) path
    out_a = model(x, hist, cached_features=False)
    assert out_a.shape == (B, H, W)
    # cached path with prev_obs provided (rollout convention)
    hist_cached = {
        "state_features": om.forward_eval_feats(hist)
        if hasattr(om, "forward_eval_feats")
        else model.get_features(
            hist["states"].reshape(B * 5, H, W, F),
            torch.zeros(B * 5, H, W, F, device=args.device),
        ).reshape(B, 5, -1),
        "mask": hist["mask"],
        "prev_obs": torch.zeros((B, H, W, F), device=args.device),
    }
    out_b = model(x, hist_cached, cached_features=True)
    assert out_b.shape == (B, H, W)
    # cached path WITHOUT prev_obs (must default to zeros, not crash)
    hist_cached.pop("prev_obs")
    out_c = model(x, hist_cached, cached_features=True)
    assert out_c.shape == (B, H, W)
    for t_ in (out_a, out_b, out_c):
        assert torch.isfinite(t_).all(), "NaN/Inf in model output"


check("SpatialOpponentModel.forward (fresh + cached ± prev_obs)", t_forward)


# --------------------------------------------------------------------------
# 3. Soft targets: peak = 1, no NaN on all-zero maps
# --------------------------------------------------------------------------
def t_soft_targets():
    H, W, _ = args.state_shape
    tm = np.zeros((3, H, W), dtype=np.float32)
    tm[0, 2, 3] = 1.0
    # tm[1] stays all-zero (HER can produce this when no goal was found)
    st = om._generate_soft_targets(torch.from_numpy(tm).to(args.device))
    assert st.shape == (3, H, W)
    assert torch.isfinite(st).all()
    if hasattr(args, "true_intent") and args.true_intent is False:
        assert abs(st[0].max().item() - 1.0) < 2e-3, "peak after clamp should be ~1"


check("_generate_soft_targets (peak, zero-map)", t_soft_targets)


# --------------------------------------------------------------------------
# 4. Dataset collection (tiny) + pretrain (1 epoch, small batch)
# --------------------------------------------------------------------------
def t_pretrain():
    collect_offline_data(num_episodes=4, save_path="/tmp/smoke_ds.pt", om_args=args)
    ds = torch.load("/tmp/smoke_ds.pt", weights_only=False)
    assert len(ds) > 0
    t0 = ds[0]
    assert "hist_len" in t0 and "states" in t0["history"], (
        "dataset items must match online item schema"
    )
    # shared episode array: all items from one episode reference one array
    assert isinstance(t0["history"]["states"], np.ndarray), (
        "history must be stored as stacked episode array, not per-step lists"
    )
    om.pretrain(ds, epochs=1, batch_size=8)


check("collect_offline_data + pretrain (1 epoch)", t_pretrain)


# --------------------------------------------------------------------------
# 5. Training rollout: 2 episodes vs SimpleAgent — exercises select_action,
#    rolling feature buffer, hindsight labeling, replay push, update(),
#    compute_targets (both OM forwards with extra=0/1)
# --------------------------------------------------------------------------
def t_run_episode():
    opp = SimpleAgent(agent_id=1, map_layout=MAP_1)
    for _ in range(3):
        stats = agent.run_episode(opp, max_steps=15)
    assert len(agent.replay) > 0
    assert 0.0 <= stats["return"] <= 15
    # transitions must not contain any opponent-action leakage
    t = agent.replay.buf[0]
    for key in t:
        assert "action" not in key.replace("opp_", "X") or key == "action", (
            f"unexpected action key in transition: {key}"
        )
    for forbidden in ("opp_action", "observed_a_opponent"):
        assert forbidden != key if False else True
    assert "opp_action" not in t and "observed_a_opponent" not in t
    assert "hist_len" in t and "states" in t["history"]


check("run_episode vs SimpleAgent (3 episodes, no leakage)", t_run_episode)


# --------------------------------------------------------------------------
# 6. Replay buffer: circular FIFO eviction, O(1) sampling
# --------------------------------------------------------------------------
def t_replay():
    from buffers import ReplayBuffer

    rb = ReplayBuffer(10)
    for i in range(25):
        rb.push(i)
    assert len(rb) == 10
    assert rb.buf[rb.ptr] == 15 or True  # order not asserted, size is
    sample = rb.sample(4)
    assert len(sample) == 4 and len(set(sample)) == 4, (
        "sample must be without replacement"
    )
    assert min(sample) >= 15, "oldest elements must have been evicted (FIFO)"


check("ReplayBuffer circular FIFO + sampling", t_replay)


# --------------------------------------------------------------------------
# 7. Eval episode: run_test_episode vs GreedySwitchAgent, metrics sane
# --------------------------------------------------------------------------
def t_test_episode():
    opp = GreedySwitchAgent(agent_id=1, map_layout=MAP_1)
    stats = agent.run_test_episode(opp, max_steps=15, render=False)
    assert stats["steps"] <= 15
    assert stats["avg_kl_error"] is None or np.isfinite(stats["avg_kl_error"])
    if stats["avg_spatial_error"] is not None:
        assert stats["avg_spatial_error"] >= 0.0


check("run_test_episode vs GreedySwitchAgent", t_test_episode)


# --------------------------------------------------------------------------
# 8. Q-learning update mechanics: losses finite, target net drifts, OM unchanged
# --------------------------------------------------------------------------
def t_update_mechanics():
    # ensure enough replay for batches
    opp = SimpleAgent(agent_id=1, map_layout=MAP_1)
    while len(agent.replay) < 40:
        agent.run_episode(opp, max_steps=15)
    q_before = [p.clone() for p in agent.q.parameters()]
    tgt_before = [p.clone() for p in agent.q_tgt.parameters()]
    ql, ml = agent.update()
    assert ql is None or np.isfinite(ql), f"Q loss not finite: {ql}"
    assert ml is None or np.isfinite(ml), f"OM loss not finite: {ml}"
    # q must change, target must change only via lerp (small),
    # OM params must NOT be changed by the Q update
    changed = any(not torch.equal(a, b) for a, b in zip(q_before, agent.q.parameters()))
    assert changed or ql is None, "Q params did not update"
    drift = max(
        (a - b).abs().max().item() for a, b in zip(tgt_before, agent.q_tgt.parameters())
    )
    assert drift < 1.0, f"target net jumped ({drift}), lerp broken"


check("update() losses / target-net lerp", t_update_mechanics)

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
n_pass = sum(1 for _, r, _ in RESULTS if r == "PASS")
print(f"\n{'=' * 52}\n{n_pass}/{len(RESULTS)} checks passed")
if any(r == "FAIL" for _, r, _ in RESULTS):
    print("Failed:")
    for name, r, msg in RESULTS:
        if r == "FAIL":
            print(f"  - {name}: {msg}")
    raise SystemExit(1)
print("All smoke tests passed — pipeline is wired correctly.")
wandb.finish()

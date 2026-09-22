"""Lambda-return correctness tests for the temporal agent (Daley & Amato 2019).

Guards the TD(lambda) target machinery added to temporal_agent.py:

  1. _compute_lambda_targets reproduces the exact forward-view lambda-return
     mixture  T_i = (1-lam) * sum_{n<N} lam^{n-1} G^(n) + lam^{N-1} G^(N)
     computed brute-force from the per-agent episode streams (mocked V's).
  2. lam_horizon truncation matches the truncated mixture (1-step bootstrap
     at the cut).
  3. The walk survives circular-buffer wraparound (FIFO contiguity).
  4. The full update() runs with lambda enabled and produces finite losses.
  5. n_step fields are forced to 1-step semantics when lam >= 0.

Run:  /opt/homebrew/anaconda3/envs/om/bin/python tests/test_lambda_returns.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import wandb
from omexplore.agents.team_agents import TeamAgent
from omexplore.agents.temporal_agent import QLearningAgent as QLearningAgentTemporal
from omexplore.envs.roadmap_foraging_env import TeamRoadmapEnv
from omexplore.models.opponent_model import OpponentModel
from omexplore.models.transformers import SpatialOpponentModel
from omexplore.utils.omg_args import OMGArgs

wandb.init(mode="disabled", project="lambda-returns-test")

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


def make_agent(team_sizes, capacity, max_steps=15, num_goals=4, **overrides):
    env = TeamRoadmapEnv(
        max_steps=max_steps, num_goals=num_goals, team_sizes=team_sizes
    )
    args = OMGArgs(
        device="cpu",
        state_shape=(env.height, env.width, env.features),
        H=env.height,
        W=env.width,
        action_dim=8,
        max_steps=max_steps,
        max_history_length=8,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        capacity=capacity,
        min_replay=10**9,  # collect only; no updates during episodes
        batch_size=8,
        train_every=1,
        gamma=0.9,
        lam=0.8,
        friendly_om=True,
        **overrides,
    )
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    hostile_om = OpponentModel(SpatialOpponentModel(args), args)
    friendly_om = OpponentModel(SpatialOpponentModel(args), args)
    agent = QLearningAgentTemporal(env, hostile_om, friendly_om, args=args)
    return agent


def collect(agent, n_episodes):
    n_opp = agent.env.team_sizes[1]
    opp = TeamAgent(agent.env, team_id=1, personas=("random",) * n_opp)
    for _ in range(n_episodes):
        agent.run_episode(opp, max_steps=agent.args.max_steps)


def v_fn(tr):
    """Deterministic mock V(s') for a transition (depends only on identity
    fields, so the brute force can recompute it from the streams)."""
    rng = (tr["episode_key"] * 131 + tr["agent_id"] * 17 + tr["step_idx"] * 7) % 97
    return (rng / 97.0) * 2.0 - 1.0


def mock_eval(transitions):
    return torch.tensor([v_fn(t) for t in transitions], dtype=torch.float32)


def streams_from_buffer(agent):
    """Rebuild the per-(episode, agent) streams straight from the buffer,
    independently of the walk logic."""
    streams = {}
    for tr in agent.replay.buf:
        if tr is None:
            continue
        streams.setdefault((tr["episode_key"], tr["agent_id"]), []).append(tr)
    for s in streams.values():
        s.sort(key=lambda t: t["step_idx"])
    return streams


def brute_force_target(stream, i, lam, gamma, horizon):
    """Closed-form (truncated) lambda-return for stream position i.

    T_i = (1-lam) * sum_{n=1}^{N-1} lam^{n-1} G^(n) + lam^{N-1} G^(N),
    with N = min(horizon or inf, L - i); G^(n) = sum_{k<n} g^k r_{i+k}
    + g^n V(s_{i+n}) and V(s_{i+n}) = v_fn(stream[i+n-1]); G^(N) at the
    episode end is the plain Monte-Carlo sum (no bootstrap).
    """
    L = len(stream)
    N = L - i
    if horizon and horizon > 0:
        N = min(N, horizon)
    total = 0.0
    for n in range(1, N + 1):
        g = sum((gamma**k) * stream[i + k]["reward"] for k in range(n))
        if n < L - i:  # bootstrap exists (i+n is a real state)
            g += (gamma**n) * v_fn(stream[i + n - 1])
        weight = (1 - lam) * (lam ** (n - 1)) if n < N else lam ** (N - 1)
        total += weight * g
    return total


def verify_all_slots(agent, horizon, label):
    agent._eval_next_values = mock_eval
    buf = agent.replay.buf
    worst = 0.0
    checked = 0
    for slot in range(agent.replay.size):
        tr = buf[slot]
        if tr is None:
            continue
        got = agent._compute_lambda_targets([slot]).item()
        stream = streams_from_buffer(agent)[(tr["episode_key"], tr["agent_id"])]
        # Position within the SURVIVING stream -- the oldest episode's head
        # may be evicted, so position != step_idx for truncated streams.
        i = next(idx for idx, t in enumerate(stream) if t["step_idx"] == tr["step_idx"])
        want = brute_force_target(stream, i, agent.args.lam, agent.args.gamma, horizon)
        worst = max(worst, abs(got - want))
        checked += 1
    assert checked > 0, "empty buffer"
    assert worst < 1e-5, f"{label}: max |err| = {worst:.2e} over {checked} slots"
    print(f"    {label}: {checked} slots verified, max |err| {worst:.2e}")
    agent._eval_next_values = QLearningAgentTemporal._eval_next_values.__get__(agent)


# --------------------------------------------------------------------------
# 1. closed form, 2v2 (stride-2 interleaved pushes), unlimited horizon
# --------------------------------------------------------------------------
def t_closed_form_2v2():
    agent = make_agent((2, 2), capacity=5000)
    collect(agent, 6)
    verify_all_slots(agent, horizon=0, label="2v2 unlimited")


# --------------------------------------------------------------------------
# 2. lam_horizon truncation
# --------------------------------------------------------------------------
def t_horizon_cut():
    agent = make_agent((1, 1), capacity=5000)
    collect(agent, 5)
    agent.args.lam_horizon = 3
    verify_all_slots(agent, horizon=3, label="horizon=3")
    agent.args.lam_horizon = 1  # degenerates to the plain 1-step DDQN target
    verify_all_slots(agent, horizon=1, label="horizon=1 (1-step)")


# --------------------------------------------------------------------------
# 3. circular-buffer wraparound
# --------------------------------------------------------------------------
def t_wraparound():
    agent = make_agent((1, 1), capacity=37)  # < 2 episodes of transitions
    collect(agent, 14)  # wraps the buffer several times
    assert agent._episode_counter == 14
    verify_all_slots(agent, horizon=0, label="wraparound")


# --------------------------------------------------------------------------
# 4. end-to-end update() with real V evaluations
# --------------------------------------------------------------------------
def t_update_end_to_end():
    agent = make_agent((1, 1), capacity=2000)
    collect(agent, 4)
    agent.args.min_replay = 10
    agent.args.train_every = 1
    losses = []
    for _ in range(12):
        agent.global_step += 1
        out = agent.update()
        q_loss = out[0]
        if q_loss is not None:
            assert np.isfinite(q_loss), f"non-finite q_loss: {q_loss}"
            losses.append(q_loss)
    assert losses, "update() never produced a loss"
    print(f"    {len(losses)} updates, last q_loss {losses[-1]:.4f}")


# --------------------------------------------------------------------------
# 5. n_step forced to 1-step semantics when lam >= 0
# --------------------------------------------------------------------------
def t_n_step_forced_to_one():
    agent = make_agent((1, 1), capacity=2000, n_step=5)
    collect(agent, 2)
    checked = 0
    for tr in agent.replay.buf:
        if tr is None or tr.get("done_n"):
            continue
        assert np.array_equal(tr["next_state_n"], tr["next_state"]), (
            "lam mode must store 1-step next_state_n (got an n-step one)"
        )
        checked += 1
    assert checked > 0
    print(f"    {checked} transitions have 1-step next_state_n under lam")


check("closed-form lambda targets (2v2, unlimited)", t_closed_form_2v2)
check("lam_horizon truncation", t_horizon_cut)
check("circular-buffer wraparound", t_wraparound)
check("end-to-end update() with lam", t_update_end_to_end)
check("n_step forced to 1 under lam", t_n_step_forced_to_one)

fails = [r for r in RESULTS if r[1] == "FAIL"]
print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} passed")
sys.exit(1 if fails else 0)

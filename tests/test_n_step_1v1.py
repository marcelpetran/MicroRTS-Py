"""Unit checks for the 1v1 QLearningAgent._add_n_step_returns (synthetic data).

Mirrors tests/test_n_step_returns.py but for the 1v1 agent's transition
schema (single flat stream, pre-augmented states instead of separate
belief fields):
  - n=3 discounted reward sums, done_n flags, bootstrap anchors
  - n=1 reproduces the 1-step semantics exactly (n_reward == reward,
    next_state_n == next_state, next_state_aug_n == next_state_aug,
    done_n == done)
  - the flat episode list is treated as ONE stream (regression guard for
    the original port bug where transitions were iterated as "streams")
  - single-transition episode edge case

Run:  python tests/test_n_step_1v1.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omexplore.agents.q_agent import QLearningAgent


def fake(n_step, gamma):
    return NS(args=NS(n_step=n_step, gamma=gamma))


# --- n=3, gamma=0.9, single stream with rewards [1, 2, 3, 4] ---
trans = []
for i in range(4):
    trans.append(
        {
            "reward": float(i + 1),
            "state": f"s{i}",
            "state_aug": f"sa{i}",
            "next_state": f"s{i + 1}",
            "next_state_aug": f"sa{i + 1}",
            "done": i == 3,
            "hist_len": i,
        }
    )
QLearningAgent._add_n_step_returns(fake(3, 0.9), trans)

exp = [1 + 0.9 * 2 + 0.81 * 3, 2 + 0.9 * 3 + 0.81 * 4, 3 + 0.9 * 4, 4.0]
for t, e in zip(trans, exp):
    assert abs(t["n_reward"] - e) < 1e-9, (t["n_reward"], e)
assert [t["done_n"] for t in trans] == [False, True, True, True]
assert trans[0]["next_state_n"] == "s3"
assert trans[0]["next_state_aug_n"] == "sa3"
assert trans[0]["hist_len_n"] == 3
assert trans[1]["next_state_n"] == "s4"
assert trans[1]["next_state_aug_n"] == "sa4"
assert trans[2]["next_state_n"] == "s4"
assert trans[3]["hist_len_n"] == 4
assert trans[3]["next_state_n"] == "s4"
# original 1-step fields must survive untouched (additive rewrite only)
for i, t in enumerate(trans):
    assert t["reward"] == float(i + 1)
    assert t["state"] == f"s{i}" and t["next_state"] == f"s{i + 1}"
    assert t["done"] == (i == 3)

# --- n=1 must reproduce the previous 1-step semantics exactly ---
trans2 = []
for i in range(3):
    trans2.append(
        {
            "reward": float(i),
            "state": f"s{i}",
            "state_aug": f"sa{i}",
            "next_state": f"s{i + 1}",
            "next_state_aug": f"sa{i + 1}",
            "done": i == 2,
            "hist_len": i,
        }
    )
QLearningAgent._add_n_step_returns(fake(1, 0.998), trans2)
for t in trans2:
    assert t["n_reward"] == t["reward"]
    assert t["next_state_n"] == t["next_state"]
    assert t["next_state_aug_n"] == t["next_state_aug"]
    assert t["done_n"] == t["done"]
    assert t["hist_len_n"] == t["hist_len"] + 1

# --- single-transition episode (immediate done) ---
trans3 = [
    {
        "reward": 5.0,
        "state": "s0",
        "state_aug": "sa0",
        "next_state": "s1",
        "next_state_aug": "sa1",
        "done": True,
        "hist_len": 0,
    }
]
QLearningAgent._add_n_step_returns(fake(3, 0.9), trans3)
assert trans3[0]["n_reward"] == 5.0
assert trans3[0]["done_n"] is True
assert trans3[0]["next_state_n"] == "s1"
assert trans3[0]["hist_len_n"] == 1

# --- classic agent (no OM): augmented-only schema, no history fields ---
from omexplore.agents.q_agent_classic import QLearningAgentClassic

ctrans = []
for i in range(4):
    ctrans.append(
        {
            "reward": float(i + 1),
            "state": f"sa{i}",  # already belief-augmented
            "next_state": f"sa{i + 1}",
            "done": i == 3,
        }
    )
QLearningAgentClassic._add_n_step_returns(fake(3, 0.9), ctrans)
for t, e in zip(ctrans, exp):
    assert abs(t["n_reward"] - e) < 1e-9, (t["n_reward"], e)
assert [t["done_n"] for t in ctrans] == [False, True, True, True]
assert ctrans[0]["next_state_n"] == "sa3"
assert ctrans[2]["next_state_n"] == "sa4"

# n=1 exact 1-step semantics for the classic agent too
ctrans2 = [
    {
        "reward": 1.0,
        "state": f"sa{i}",
        "next_state": f"sa{i + 1}",
        "done": i == 2,
    }
    for i in range(3)
]
QLearningAgentClassic._add_n_step_returns(fake(1, 0.99), ctrans2)
for t in ctrans2:
    assert t["n_reward"] == t["reward"]
    assert t["next_state_n"] == t["next_state"]
    assert t["done_n"] == t["done"]

print("all 1v1 n-step unit checks passed")

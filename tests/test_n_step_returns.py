"""Quick unit checks for QLearningAgent._add_n_step_returns (synthetic data)."""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omexplore.agents.q_agent_team import QLearningAgent


def fake(n_step, gamma):
    return NS(args=NS(n_step=n_step, gamma=gamma))


# --- n=3, gamma=0.9, single agent stream with rewards [1, 2, 3, 4] ---
trans = []
for i in range(4):
    trans.append(
        {
            "agent_id": 0,
            "reward": float(i + 1),
            "state": f"s{i}",
            "belief": f"b{i}",
            "next_state": f"s{i + 1}",
            "next_belief": f"b{i + 1}",
            "done": i == 3,
            "hist_len": i,
        }
    )
QLearningAgent._add_n_step_returns(fake(3, 0.9), trans)

exp = [1 + 0.9 * 2 + 0.81 * 3, 2 + 0.9 * 3 + 0.81 * 4, 3 + 0.9 * 4, 4.0]
for t, e in zip(trans, exp):
    assert abs(t["n_reward"] - e) < 1e-9, (t["n_reward"], e)
assert [t["done_n"] for t in trans] == [False, True, True, True]
assert trans[0]["next_state_n"] == "s3" and trans[0]["hist_len_n"] == 3
assert trans[1]["next_state_n"] == "s4" and trans[1]["next_belief_n"] == "b4"
assert trans[2]["next_state_n"] == "s4" and trans[3]["hist_len_n"] == 4

# --- n=1 must reproduce the previous 1-step semantics exactly ---
trans2 = []
for i in range(3):
    trans2.append(
        {
            "agent_id": 7,
            "reward": float(i),
            "state": f"s{i}",
            "belief": f"b{i}",
            "next_state": f"s{i + 1}",
            "next_belief": f"b{i + 1}",
            "done": i == 2,
            "hist_len": i,
        }
    )
QLearningAgent._add_n_step_returns(fake(1, 0.998), trans2)
for t in trans2:
    assert t["n_reward"] == t["reward"]
    assert t["next_state_n"] == t["next_state"]
    assert t["next_belief_n"] == t["next_belief"]
    assert t["done_n"] == t["done"]
    assert t["hist_len_n"] == t["hist_len"] + 1

# --- interleaved agents must be grouped into independent streams ---
mix = []
for i in range(3):
    for a in (0, 1):
        mix.append(
            {
                "agent_id": a,
                "reward": 1.0,
                "state": f"s{a}{i}",
                "belief": "b",
                "next_state": f"s{a}{i + 1}",
                "next_belief": "b",
                "done": i == 2,
                "hist_len": i,
            }
        )
QLearningAgent._add_n_step_returns(fake(2, 0.5), mix)
# per stream (L=3, n=2): i=0 -> 1+0.5*1, i=1 -> 1+0.5*1, i=2 -> 1.0
for t in mix:
    i = t["hist_len"]
    expected = 1.5 if i < 2 else 1.0
    assert abs(t["n_reward"] - expected) < 1e-9, t
assert mix[0]["next_state_n"] == "s02"  # agent 0, i=0 -> its own t+2 state

print("all n-step unit checks passed")

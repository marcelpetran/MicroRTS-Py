"""Validation ladder step 1: does fog-greedy THROUGH the roadmap graph
preserve the grid fog-greedy return?

Runs, on identical episode layouts (paired seeds):

  --policy grid  : TeamAgent(greedy, know_all_goals) on both teams - the
                   reference that produced the ~8.81 ceiling on tiny11.
  --policy graph : team 0 = graph fog-greedy (pick nearest believed goal,
                   commit to the roadmap node nearest that goal, walk via
                   BFS, re-decide on the wrapper's triggers), team 1 =
                   the same greedy TeamAgent as above.

Reports mean team scores, decisions per episode, mean decision length k,
and the trigger histogram for the graph policy.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from omexplore.agents.team_agents import TeamAgent
from omexplore.envs.roadmap_decision_env import RoadmapDecisionEnv
from omexplore.envs.roadmap_foraging_env import TeamRoadmapEnv
from omexplore.envs.roadmap_graph import RoadmapGraph, list_roadmaps


def find_roadmap(map_name, explicit=None):
    if explicit:
        return explicit
    cands = [p for p in list_roadmaps(map_name) if "_w" not in Path(p).stem]
    if not cands:
        raise SystemExit(f"no .roadmap found for map {map_name}; pass --roadmap")
    cands.sort(key=lambda p: Path(p).stat().st_size)
    print(f"using roadmap: {cands[0]}")
    return cands[0]


class GraphFogGreedy:
    """Nearest believed goal -> best waypoint node; commit the goal.

    node = argmin_n ( BFS(me -> n) + BFS(n -> goal) ): the roadmap node
    that best keeps the walk on the way to the goal (waypoint scoring),
    instead of the node nearest to the goal which can force detours.
    Pass --node_score nearest for the old behaviour.
    """

    def __init__(self, denv: RoadmapDecisionEnv, seed=None, node_score="waypoint"):
        self.denv = denv
        self.rng = np.random.default_rng(seed)
        self.n_nodes = denv.graph.n_nodes
        self.node_score = node_score

    def decide(self, aid, taken=frozenset()):
        env, graph = self.denv.env, self.denv.graph
        pos = tuple(env.agents[aid])
        goals = sorted(self.denv.belief.belief_food - set(taken))
        if not goals:  # wander: random node (mirrors GreedyMember's random walk)
            return int(self.rng.integers(self.n_nodes)), None
        fpos = env._dist_field(pos)
        g = min(
            goals,
            key=lambda p: fpos[p[0], p[1]] if fpos[p[0], p[1]] >= 0 else np.inf,
        )
        fg = env._dist_field(g)
        to_nodes = fpos[graph.node_cells[:, 0], graph.node_cells[:, 1]]
        from_goal = fg[graph.node_cells[:, 0], graph.node_cells[:, 1]]
        to_nodes = np.where(to_nodes >= 0, to_nodes, np.inf)
        from_goal = np.where(from_goal >= 0, from_goal, np.inf)
        if self.node_score == "waypoint":
            score = to_nodes + from_goal
        else:  # "nearest": node closest to the goal
            score = from_goal
        node = int(np.argmin(score))
        return node, tuple(g)


def run_graph(args):
    env = TeamRoadmapEnv(
        map_name=args.map,
        max_steps=args.max_steps,
        vision_radius=args.vision_radius,
        num_goals=args.num_goals,
        team_sizes=tuple(int(x) for x in args.team_sizes.split(",")),
    )
    graph = RoadmapGraph(args.roadmap)
    opp = TeamAgent(env, team_id=1, personas=("greedy",) * env.team_sizes[1])
    denv = RoadmapDecisionEnv(
        env,
        graph,
        learner_team=0,
        opp_action_fn=opp.select_actions,
        decision_timeout=args.decision_timeout,
        opp_trigger=args.opp_trigger,
        walk_commit=args.walk_commit,
        walk_mode=args.walk_mode,
        seed=args.seed,
    )
    policy = GraphFogGreedy(denv, seed=args.seed, node_score=args.node_score)
    # Explicit policy commits (with batch `taken` coordination) are always
    # passed: the wrapper prefers them in both walk modes; the region
    # derivation is the NN-facing fallback when no commit is given.

    s0, s1 = [], []
    dec_count, k_total, trig = [], 0, {}
    for ep in range(args.episodes):
        denv.reset(seed=args.seed + ep)
        opp.reset()
        done = False
        n_dec = 0
        redecide = set(denv.learn_ids)
        while not done:
            decisions, commits = {}, {}
            taken = set()
            for aid in sorted(redecide):
                node, goal = policy.decide(aid, taken)
                decisions[aid] = node
                if goal is not None:
                    taken.add(goal)
                    commits[aid] = goal
            obs, r, done, info = denv.step(decisions, commits)
            n_dec += len(decisions)
            k_total += info["k"]
            for t in info["triggers"].values():
                trig[t] = trig.get(t, 0) + 1
            redecide = info["redecide"]
        dec_count.append(n_dec)
        s0.append(env.team_scores[0])
        s1.append(env.team_scores[1])
    return s0, s1, dec_count, k_total, trig


def run_grid(args):
    env = TeamRoadmapEnv(
        map_name=args.map,
        max_steps=args.max_steps,
        vision_radius=args.vision_radius,
        num_goals=args.num_goals,
        team_sizes=tuple(int(x) for x in args.team_sizes.split(",")),
    )
    cache = {}
    t0 = TeamAgent(
        env, team_id=0, personas=("greedy",) * env.team_sizes[0], field_cache=cache
    )
    t1 = TeamAgent(
        env, team_id=1, personas=("greedy",) * env.team_sizes[1], field_cache=cache
    )

    s0, s1 = [], []
    for ep in range(args.episodes):
        np.random.seed(args.seed + ep)
        obs = env.reset()
        t0.reset()
        t1.reset()
        done = False
        while not done:
            actions = {**t0.select_actions(obs), **t1.select_actions(obs)}
            obs, _, done, _ = env.step(actions)
        s0.append(env.team_scores[0])
        s1.append(env.team_scores[1])
    return s0, s1, None, None, None


def report(name, s0, s1, dec_count, k_total, trig, episodes):
    s0, s1 = np.array(s0), np.array(s1)
    print(f"\n=== {name} ({episodes} episodes) ===")
    print(
        f"team0 (policy)  mean {s0.mean():.3f}  std {s0.std():.3f}  "
        f"95%CI ±{1.96 * s0.std() / np.sqrt(len(s0)):.3f}"
    )
    print(
        f"team1 (greedy)  mean {s1.mean():.3f}  std {s1.std():.3f}  "
        f"95%CI ±{1.96 * s1.std() / np.sqrt(len(s1)):.3f}"
    )
    if dec_count is not None:
        dc = np.array(dec_count)
        print(
            f"decisions/episode: mean {dc.mean():.1f}  min {dc.min()}  max {dc.max()}"
        )
        print(f"mean decision length k: {k_total / max(1, dc.sum()):.2f}")
        print(f"triggers: {dict(sorted(trig.items(), key=lambda x: -x[1]))}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--map", default="tiny11")
    p.add_argument("--roadmap", default=None)
    p.add_argument("--team_sizes", default="1,1")
    p.add_argument("--num_goals", type=int, default=11)
    p.add_argument("--vision_radius", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=50)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--decision_timeout", type=int, default=None)
    p.add_argument("--opp_trigger", choices=["appear", "move", "off"], default="appear")
    p.add_argument("--node_score", choices=["waypoint", "nearest"], default="nearest")
    p.add_argument(
        "--walk_commit",
        action="store_true",
        help="diagnostic: walk straight to the committed goal instead of the node",
    )
    p.add_argument("--walk_mode", choices=["node", "region"], default="region")
    p.add_argument("--policy", choices=["grid", "graph", "both"], default="both")
    a = p.parse_args()
    a.roadmap = find_roadmap(a.map, a.roadmap)

    if a.policy in ("grid", "both"):
        report("GRID fog-greedy", *run_grid(a), a.episodes)
    if a.policy in ("graph", "both"):
        report("GRAPH fog-greedy", *run_graph(a), a.episodes)


if __name__ == "__main__":
    main()

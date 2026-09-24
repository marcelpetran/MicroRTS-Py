"""GNN input construction for the roadmap-based exploration agent.

Consumes a RoadmapDecisionEnv observation snapshot and produces the
per-node feature matrix, (static) edge structure, and global features
for the S2V-style GNN. See the design discussion: the roadmap graph is
STATIC for a whole episode - only node features x and global features g
change between decisions, so edge_index/edge_weight are built once.

Hard constraints (each grounded in a failure we already hit once):
  - edges must be SYMMETRIZED (roadmap files store each edge once)
  - BFS fields use -1 for unreachable - mask before aggregating
  - empty belief sets need sentinel + flag columns, not zeros
    (the belief-ignored failure of the CNN probe)
  - features come from BELIEF + team vision + own position only,
    never from env.food_positions / true opponent positions
  - distance features normalized log1p(d) / log1p(max_steps)
"""

from __future__ import annotations

import numpy as np

from omexplore.envs.roadmap_decision_env import RoadmapDecisionEnv

# Column layout of x. Keep in sync with the documentation table.
NODE_FEATURES = [
    "my_eta",  # 0: log-normalized BFS from deciding agent to node
    "goal_eta",  # 1: min BFS node -> believed goal (sentinel if none)
    "goal_known",  # 2: flag, any believed goal within region radius
    "goal_count",  # 3: believed goals within region radius, capped/5
    "opp_eta",  # 4: min BFS node -> believed opponent (sentinel+flag)
    "is_visible",  # 5: node cell inside current team vision
    "am_here",  # 6: deciding agent inside this node's region
    "opp_vis_eta",  # 7: min BFS node -> currently visible opponent cell
]
N_NODE_FEATURES = len(NODE_FEATURES)

GLOBAL_FEATURES = [
    "time_left",  # 0: 1 - step / max_steps
    "goals_believed",  # 1: believed-uncollected goals / initial goal count
    "score_diff",  # 2: (my team - best other team) / initial goal count
    "opp_age",  # 3: steps since opponent last seen / max_steps
    "coverage",  # 4: team coverage fraction (env.get_coverage)
]
N_GLOBAL_FEATURES = len(GLOBAL_FEATURES)


class GraphFeatureBuilder:
    """Static graph + dynamic egocentric node features -> GNN input dict.

    build(aid) returns:
      x           (N, N_NODE_FEATURES) float32
      edge_index  (2, E) int64, [src; dst], both directions
      edge_weight  (E,) float32, normalized BFS steps
      g            (N_GLOBAL_FEATURES,) float32
    """

    def __init__(self, denv: RoadmapDecisionEnv):
        self.denv = denv
        self.graph = denv.graph
        self.env = denv.env
        ei = self.graph.edge_index
        ew = self.graph.edge_weight
        src = np.concatenate([ei[:, 0], ei[:, 1]])
        dst = np.concatenate([ei[:, 1], ei[:, 0]])
        self.edge_index = np.stack([src, dst], axis=0)
        self.edge_weight = np.concatenate([ew, ew])
        self.max_steps = self.env.max_steps
        self.sentinel_norm = np.float32(
            np.log1p(2 * self.max_steps) / np.log1p(self.max_steps)
        )
        w_norm = np.log1p(self.edge_weight) / np.log1p(self.max_steps)
        self.edge_weight = w_norm.astype(np.float32)
        # per node radius
        self.region_radius = np.zeros(self.graph.n_nodes, dtype=np.float32)
        for i in range(self.denv.graph.n_nodes):
            mask = self.denv.graph.region == i
            if not mask.any():
                self.region_radius[i] = 0.0
                continue
            vals = self.denv.graph.dist_fields[i][mask]
            self.region_radius[i] = vals.max()
        self.dist_fields_flat = self.graph.dist_fields.reshape(
            self.graph.n_nodes, -1
        )  # (N, H*W)
        self.node_cells_flat = (
            self.graph.node_cells[:, 0] * self.graph.width + self.graph.node_cells[:, 1]
        )

    # ------------------------------------------------------------------ #
    def build(self, aid: int) -> dict:
        """Features for one deciding agent at the current decision point. All features are egocentric and normalized."""
        snap = self.denv.observation()
        my_team_id = self.env.teams[aid]

        # "How many steps until I could be standing there"
        my_eta = self._min_eta_to_set({snap["pos"][aid]})
        # "Am I already inside this node's region?"
        am_here = (
            self.graph.region[snap["pos"][aid]] == np.arange(self.graph.n_nodes)
        ).astype(np.float32)

        D = self._dists_matrix(snap["belief_food"])

        # “From this node, how far is the nearest believed-uncollected goal?” — global min, no radius mask.
        goal_eta = self._min_eta_to_set(snap["belief_food"])

        # Empty belief_food: goal_eta goes through the guarded _min_eta_to_set
        # (sentinel_norm). goal_known/goal_count are safe without a branch because
        # any/sum have identities (False/0) and reduce (N, 0) correctly.
        # WARNING: do NOT add a min reduction over D here — (N, 0) has no identity

        # “Is one of those goals actually inside this node’s region (within region_radius)?”
        within = D <= self.region_radius[:, None]
        goal_known = within.any(axis=1).astype(np.float32)

        # “How many of those goals are inside this node’s region (within region_radius)?”
        goal_count = np.minimum(within.sum(axis=1), 5) / 5.0

        # "How close is this node to where we believe the opponent is?"
        opp_eta = self._min_eta_to_set(snap["belief_opp"])

        # "How close is this node to where we currently know the opponent is?"
        opp_vis_eta = self._min_eta_to_set(snap["opp_vis"])

        # "Is this node's cell inside our current team vision?"
        if not snap["obs"]:
            is_visible = np.zeros(self.graph.n_nodes, dtype=np.float32)
        else:
            is_visible = snap["obs"][aid][:, :, 6][
                self.graph.node_cells[:, 0], self.graph.node_cells[:, 1]
            ].astype(np.float32)

        # Global features
        # How much of the episde remains?
        time_left = 1.0 - snap["step"] / self.env.max_steps
        # How many goals do we believe are still uncollected?
        goals_believed = len(snap["belief_food"]) / self.denv.initial_goal_count
        # "Are we ahead or behind the best opponent team?"
        team_ids = set(self.env.teams.values())
        score_diff = (
            self.env.team_scores[my_team_id]
            - max(self.env.team_scores[tid] for tid in team_ids if tid != my_team_id)
        ) / self.denv.initial_goal_count
        # "How stale is our opponent belief?
        opp_age = min(self.denv.belief.opp_age / self.env.max_steps, 1.0)
        # "How much of the map is currently covered by our team?"
        coverage = self.env.get_coverage(my_team_id)
        g = np.array(
            [time_left, goals_believed, score_diff, opp_age, coverage], dtype=np.float32
        )
        cols = [
            my_eta,
            goal_eta,
            goal_known,
            goal_count,
            opp_eta,
            is_visible,
            am_here,
            opp_vis_eta,
        ]
        x = np.stack(cols, axis=1).astype(np.float32)
        assert x.shape == (self.graph.n_nodes, N_NODE_FEATURES)
        return {
            "x": x,
            "edge_index": self.edge_index,  # (2, E) int64
            "edge_weight": self.edge_weight,  # (E,) float32
            "g": g,  # (5,) float32
        }

    # ------------------------------------------------------------------ #
    # Helper functions
    def _log_norm(self, d: np.ndarray) -> np.ndarray:
        """log1p(d) / log1p(max_steps); d may contain a sentinel."""
        return np.log1p(d) / np.log1p(self.max_steps)

    def _min_eta_to_set(self, cells) -> np.ndarray:
        """(N,) log-normalized min BFS node->cell; sentinel if set empty."""
        if not cells:
            return np.full(self.graph.n_nodes, self.sentinel_norm, dtype=np.float32)
        return self._log_norm(self._dists_matrix(cells).min(axis=1))

    def _dists_matrix(self, cells) -> np.ndarray:
        """(N, G) raw BFS steps node->cell; unreachable -> 2*max_steps; empty -> (N, 0)."""
        if not cells:
            return np.full(
                (self.graph.n_nodes, 0), 2 * self.max_steps, dtype=np.float32
            )
        flat = np.array([r * self.graph.width + c for r, c in sorted(cells)])
        return np.where(
            self.dist_fields_flat[:, flat] >= 0,
            self.dist_fields_flat[:, flat],
            2 * self.max_steps,
        )


if __name__ == "__main__":
    from omexplore.agents.team_agents import TeamAgent
    from omexplore.envs.roadmap_decision_env import RoadmapDecisionEnv
    from omexplore.envs.roadmap_foraging_env import TeamRoadmapEnv
    from omexplore.envs.roadmap_graph import RoadmapGraph

    g = RoadmapGraph("cmapf_unified/viz/tiny11_N20_K4.roadmap")
    env = TeamRoadmapEnv(
        map_name="tiny11", max_steps=50, vision_radius=3, num_goals=6, team_sizes=(2, 2)
    )
    opp = TeamAgent(env, team_id=1, personas=("greedy",) * env.team_sizes[1])
    denv = RoadmapDecisionEnv(
        env,
        g,
        learner_team=0,
        opp_action_fn=opp.select_actions,
        decision_timeout=8,
        opp_trigger="off",
    )
    _ = denv.reset()
    b = GraphFeatureBuilder(denv)
    assert b.edge_index.shape == (2, g.edge_index.shape[0] * 2)
    assert b.edge_weight.shape == (b.edge_index.shape[1],)
    assert (b.region_radius >= 0).all()

    features = b.build(aid=0)
    assert features["x"].shape == (g.n_nodes, N_NODE_FEATURES)
    assert features["edge_index"].shape == (2, g.edge_index.shape[0] * 2)
    assert features["edge_weight"].shape == (b.edge_index.shape[1],)
    assert features["g"].shape == (N_GLOBAL_FEATURES,)

    # single-goal recoverability: nearest node to the one believed goal
    denv.reset()
    denv.belief.belief_food = {(5, 5)}
    x = b.build(0)["x"]

    gk, ge = x[:, 2], x[:, 1]
    owner = g.nearest_node((5, 5))  # 4
    assert gk[owner] == 1.0  # the owner must know its own goal
    assert int(np.argmin(ge)) == owner  # owner is the nearest node

    # every flagged node genuinely within its radius — ground truth from dist_fields
    for n in np.where(gk > 0)[0]:
        d = g.dist_fields[n][5, 5]
        assert 0 <= d <= b.region_radius[n]
    assert (
        np.argmax(x[:, NODE_FEATURES.index("goal_known")])
        in (np.argsort(x[:, NODE_FEATURES.index("goal_eta")])[:3])
    )  # top-3 nearest nodes know the goal
    assert x[:, NODE_FEATURES.index("goal_known")].sum() >= 1.0

    # empty belief: sentinel + zero flags, no crash
    denv.belief.belief_food = frozenset()
    xe = b.build(0)["x"]
    assert (xe[:, 1] == b.sentinel_norm).all() and (xe[:, 2] == 0).all()

    # egocentricity: my_eta minimal at the agent's own region
    denv.reset()
    x = b.build(0)["x"]
    pos = denv.observation()["pos"][0]
    assert x[np.argmin(x[:, 0]), NODE_FEATURES.index("am_here")] == 1.0
    # global features: time_left, goals_believed, score_diff, opp_age, coverage
    gv = b.build(0)["g"]
    assert gv[0] == 1.0  # time_left at step 0
    assert gv[1] == 1.0  # all goals believed
    assert gv[2] == 0.0  # scores level
    assert gv[3] == 0.0  # opp_age fresh (prior counts as sighting)

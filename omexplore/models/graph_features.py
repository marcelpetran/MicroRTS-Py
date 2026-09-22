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
        # TODO(you): precompute static parts
        #   - symmetrized edge_index / edge_weight (files store one direction)
        #   - distance normalization scale (max_steps)
        #   - region radius for goal_known / goal_count
        #   - node cell index arrays for fancy indexing
        # graph.edge_index is (E, 2) with one direction per edge; we need both directions
        ei = self.graph.edge_index
        ew = self.graph.edge_weight
        src = np.concatenate([ei[:, 0], ei[:, 1]])
        dst = np.concatenate([ei[:, 1], ei[:, 0]])
        self.edge_index = np.stack([src, dst], axis=0)
        self.edge_weight = np.concatenate([ew, ew])
        self.max_steps = self.env.max_steps
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
        """Features for one deciding agent at the current decision point."""
        # TODO(you): implement, in this order
        #   1. my_eta      from env._dist_field(pos) at node_cells
        #   2. goal_eta    min over belief_food of graph.dist_fields[:, g]
        #                  (mask -1; sentinel+flag when belief empty)
        #   3. goal_count  belief_food within region radius of the node
        #   4. opp_eta     min over belief_opp of graph.dist_fields[:, b]
        #   5. is_visible  obs[anchor][:, :, 6] at node cells
        #   6. am_here     graph.region[pos] == node
        #   7. opp_vis_eta min over denv-visible opp cells
        #   8. global g    (time, goals, score, opp_age, coverage)
        snap = self.denv.observation()
        pos = snap["pos"][aid]

    # ------------------------------------------------------------------ #
    # Helper suggestions (add/remove freely):
    #
    # def _log_norm(self, d: np.ndarray) -> np.ndarray:
    #     """log1p(d) / log1p(max_steps); d may contain a sentinel."""
    #
    # def _min_dist_to_set(self, cells: set) -> np.ndarray:
    #     """(N,) min BFS node->cell over a cell set; sentinel if empty."""
    #
    # def _obs(self):
    #     """anchor obs from the last observation dict."""


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
    b = GraphFeatureBuilder(denv)
    assert b.edge_index.shape == (2, g.edge_index.shape[0] * 2)
    assert b.edge_weight.shape == (b.edge_index.shape[1],)
    assert (b.region_radius >= 0).all()

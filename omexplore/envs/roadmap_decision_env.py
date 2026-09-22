"""SMDP decision wrapper over TeamRoadmapEnv.

One decision = "which roadmap node to target next". A BFS low-level
controller walks each controlled agent toward its chosen node (grid-optimal
path, exact env movement semantics, no corner cutting). Decisions are
per-agent: each controlled agent re-decides when ITS trigger fires, while
the others keep executing their committed paths.

Re-decision triggers (user's four pillars), checked after every low-level
step:

  arrive     the agent exhausted its path (reached its node / goal)
  collected  the learner team collected a goal (team-wide: all re-decide)
  goal_gone  the agent's committed goal cell left belief_food (observed
             empty or collected elsewhere) - per-agent
  opp        the set of opponent cells visible to the team changed
             (team-wide)
  timeout    steps since this agent's last decision >= decision_timeout
  done       episode terminal (team-wide)

The wrapper owns the learner team's BeliefTracker (seeded with the true
goal set at reset for fairness with know_all_goals heuristic opponents)
and updates it after every low-level step from the anchor agent's
team-pooled observation.

step() runs low-level steps until at least one controlled agent must
re-decide (or the episode ends) and reports the set in info["redecide"].
Agents keep executing a committed path across the calls of other agents'
decisions; an agent whose path is exhausted but whose re-decision is not
yet requested idles (no-op) - the loop exits for it at the top of the
next step() call at the latest.

The NN-facing observation (node/edge/global features) is deliberately NOT
built here; see the feature builder module. This wrapper exposes the raw
ingredients (env, graph, belief, positions) so both scripted baselines and
learned policies can consume them.
"""

from __future__ import annotations

import numpy as np

from omexplore.envs.roadmap_graph import RoadmapGraph, bfs_field
from omexplore.models.beliefs import BeliefTracker


class RoadmapDecisionEnv:
    """Decision-level (SMDP) environment for one learning team.

    Parameters
    ----------
    env : TeamRoadmapEnv
        The low-level environment (shared with the opposing team).
    graph : RoadmapGraph
        The precomputed roadmap backbone.
    learner_team : int
        Team id controlled by the learning policy.
    opp_action_fn : callable(obs_dict) -> {agent_id: action}
        Drives all non-controlled agents every low-level step
        (e.g. TeamAgent.select_actions).
    opp_trigger : "appear" | "move" | "off"
        When the opponent-visibility trigger fires: "appear" only on
        empty->nonempty (newly spotted), "move" on any change of the
        visible opponent cell set, "off" never. "appear" is the default:
        "move" degenerates to per-step re-decisions on small maps where
        foes are almost always visible.
    walk_commit : bool
        Diagnostic: walk straight to the committed goal cell instead of
        the chosen node (keeps the trigger structure, removes the node
        detour). Used to decompose the graph-vs-grid gap into trigger
        cost vs detour cost.
    walk_mode : "node" | "region"
        Controller target after a node decision. "node": walk to the
        node cell. "region" (default): walk to the believed-uncollected
        goal nearest to the chosen node if one lies within region_radius
        of it (the node's "region intent"), else to the node cell
        (exploration). Keeps the action space = nodes while removing the
        node detour: validation on tiny11 showed the trigger structure
        preserves grid-greedy return exactly when the walk goes to the
        goal cell.
    decision_timeout : int | None
        Max low-level steps between two decisions of the same agent.
    seed : int | None
        Seed for the fallback random walk (unreachable nodes).
    """

    def __init__(
        self,
        env,
        graph: RoadmapGraph,
        learner_team: int = 0,
        opp_action_fn=None,
        decision_timeout: int | None = None,
        opp_trigger: str = "appear",
        walk_commit: bool = False,
        walk_mode: str = "region",
        seed: int | None = None,
    ):
        self.env = env
        self.graph = graph
        self.team = learner_team
        self.opp_action_fn = opp_action_fn
        self.decision_timeout = decision_timeout
        if opp_trigger not in ("appear", "move", "off"):
            raise ValueError(f"unknown opp_trigger {opp_trigger!r}")
        if walk_mode not in ("node", "region"):
            raise ValueError(f"unknown walk_mode {walk_mode!r}")
        self.opp_trigger = opp_trigger
        self.walk_commit = walk_commit
        self.walk_mode = walk_mode
        # per-node BFS distance to every cell is precomputed (dist_fields);
        # region intent lookup = min over believed goals of dist_fields[n].
        self.region_radius: int | None = None  # set by configure_region()
        self.rng = np.random.default_rng(seed)

        self.learn_ids = list(env.get_team_members(learner_team))
        self.anchor = self.learn_ids[0]
        self.belief = BeliefTracker(
            env.height, env.width, channels=(1, 4, 6), horizon=env.max_steps
        )
        # per-agent controller state
        self.paths: dict[int, list[int]] = {}
        self.commit: dict[int, tuple | None] = {}
        self.decision_age: dict[int, int] = {}
        self._prev_opp_vis: frozenset = frozenset()
        self._last_obs: dict | None = None
        self.trigger_counts: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None):
        """Reset env + belief; returns the raw per-agent obs dict."""
        if seed is not None:
            np.random.seed(seed)
        obs = self.env.reset()
        self._last_obs = obs
        # Fair prior: the learner knows the sampled goal set, exactly like
        # know_all_goals heuristic opponents (scenario knowledge).
        self.belief.set_food_prior(self.env.food_positions)
        self.belief.reset(use_map_prior=True)
        self.paths = {}
        self.commit = {}
        self.decision_age = {aid: 0 for aid in self.learn_ids}
        self._prev_opp_vis = self._opp_vis()
        self.trigger_counts = {}
        return self.observation()

    # ------------------------------------------------------------------ #
    def step(self, decisions: dict[int, int], commits: dict | None = None):
        """Commit agents to nodes and run low-level steps.

        decisions : {agent_id: node_id} for agents (re-)deciding now.
        commits   : optional {agent_id: goal_cell} - the believed goal the
                    decision was based on. If the agent is already at the
                    chosen node, it instead walks directly to the committed
                    goal cell (avoids zero-progress node picks); the goal
                    also arms the goal_gone trigger.

        Returns (obs, reward, done, info):
          reward : {agent_id: summed env reward over the k low-level steps}
          info   : {"redecide": set(agent_id), "triggers": {aid: reason},
                    "k": low-level steps, "collected": [...], "rewards": {...},
                    "team_scores": {...}}
        """
        commits = commits or {}
        for aid, node in decisions.items():
            if aid not in self.learn_ids:
                raise ValueError(f"agent {aid} is not controlled (team {self.team})")
            if not (0 <= node < self.graph.n_nodes):
                raise ValueError(f"node {node} out of range")
            self._set_target(aid, node, commits.get(aid))
            self.decision_age[aid] = 0

        reward = {aid: 0.0 for aid in self.learn_ids}
        triggers: dict[int, str] = {}
        collected_all: list = []
        k = 0

        while True:
            # --- exit conditions checked BEFORE stepping ---
            if self.env.terminal:
                triggers = {aid: "done" for aid in self.learn_ids}
                break
            arrived = [aid for aid in self.learn_ids if not self.paths.get(aid)]
            if arrived:
                for aid in arrived:
                    triggers.setdefault(aid, "arrive")
                break
            if self.decision_timeout is not None and any(
                self.decision_age[aid] >= self.decision_timeout
                for aid in self.learn_ids
            ):
                for aid in self.learn_ids:
                    if self.decision_age[aid] >= self.decision_timeout:
                        triggers.setdefault(aid, "timeout")
                break

            # --- one low-level step ---
            actions = {}
            for aid in self.learn_ids:
                actions[aid] = self.paths[aid].pop(0)
            if self.opp_action_fn is not None:
                actions.update(self.opp_action_fn(self._last_obs))
            obs, rewards, done, info = self.env.step(actions)
            self._last_obs = obs
            k += 1
            for aid in self.learn_ids:
                reward[aid] += rewards.get(aid, 0.0)
                self.decision_age[aid] += 1
            collected_all.extend(info.get("collected", []))
            self.belief.update(obs[self.anchor])

            # --- triggers checked AFTER stepping ---
            if done:
                triggers = {aid: "done" for aid in self.learn_ids}
                break
            # learner team collected a goal (team knows: it got paid)
            if any(
                a in self.learn_ids
                for collectors in info.get("collectors", {}).values()
                for a in collectors
            ):
                triggers = {aid: "collected" for aid in self.learn_ids}
                break
            # opponent visibility trigger (team-wide)
            opp_vis = self._opp_vis()
            opp_event = (
                self.opp_trigger == "move" and opp_vis != self._prev_opp_vis
            ) or (self.opp_trigger == "appear" and not self._prev_opp_vis and opp_vis)
            if opp_event:
                triggers = {aid: "opp" for aid in self.learn_ids}
                self._prev_opp_vis = opp_vis
                break
            self._prev_opp_vis = opp_vis
            # committed goal disproven (per-agent)
            gone = [
                aid
                for aid in self.learn_ids
                if self.commit.get(aid) is not None
                and self.commit[aid] not in self.belief.belief_food
            ]
            for aid in gone:
                triggers.setdefault(aid, "goal_gone")
            if triggers:
                break

        for t in triggers.values():
            self.trigger_counts[t] = self.trigger_counts.get(t, 0) + 1
        redecide = set(triggers)
        # strip paths of agents that must re-decide
        for aid in redecide:
            self.paths.pop(aid, None)
            self.commit.pop(aid, None)
        info = {
            "redecide": redecide,
            "triggers": triggers,
            "k": k,
            "collected": collected_all,
            "rewards": reward,
            "team_scores": dict(self.env.team_scores),
        }
        return self.observation(), reward, self.env.terminal, info

    # ------------------------------------------------------------------ #
    def observation(self) -> dict:
        """Raw decision-level snapshot (feature builder consumes this)."""
        return {
            "obs": self._last_obs,
            "pos": {aid: tuple(self.env.agents[aid]) for aid in self.learn_ids},
            "belief_food": frozenset(self.belief.belief_food),
            "belief_opp": frozenset(self.belief.belief_opp),
            "opp_vis": self._prev_opp_vis,
            "step": self.env.steps,
            "redecide": frozenset(
                aid for aid in self.learn_ids if not self.paths.get(aid)
            ),
        }

    # ------------------------------------------------------------------ #
    def _set_target(self, aid: int, node: int, commit_goal=None):
        pos = self.env.agents[aid]
        if self.walk_mode == "region" or self.walk_commit:
            # Region intent: the believed-uncollected goal nearest to the
            # chosen node (within its region) becomes the walk target and
            # the commit; the node cell is only an exploration fallback.
            if commit_goal is None:
                commit_goal = self._region_goal(node)
            if commit_goal is not None:
                path = self._actions_to(tuple(pos), tuple(commit_goal))
                if path is not None:
                    self.paths[aid] = list(path)
                    self.commit[aid] = tuple(commit_goal)
                    return
        path = self.graph.path_actions(tuple(pos), node)
        if path is None:  # unreachable (should not happen): random walk
            path = [int(self.rng.integers(8))] * 4
            self.commit[aid] = None
            self.paths[aid] = path
            return
        if (
            not path
            and commit_goal is not None
            and tuple(commit_goal) != tuple(self.graph.node_cells[node])
        ):
            # Zero-progress node pick: walk directly to the committed goal.
            path = self._actions_to(tuple(pos), tuple(commit_goal)) or [
                int(self.rng.integers(8))
            ]
        self.paths[aid] = list(path)
        self.commit[aid] = tuple(commit_goal) if commit_goal is not None else None

    def _region_goal(self, node: int, radius: int | None = None):
        """Believed-uncollected goal nearest to `node`, if within radius.

        Uses the precomputed per-node BFS distance fields, so this is a
        cheap scan over the current belief set. radius defaults to the
        roadmap's maximum region radius + 1 (a goal in another node's
        region is that node's intent, not this one's).
        """
        if not self.belief.belief_food:
            return None
        if radius is None:
            if self.region_radius is None:
                # farthest distance from any free cell to its region's node
                reg = self.graph.region
                rows, cols = np.nonzero(reg >= 0)
                d = self.graph.dist_fields[reg[rows, cols], rows, cols]
                self.region_radius = int(d.max()) + 1
            radius = self.region_radius
        field = self.graph.dist_fields[node]
        best, best_d = None, None
        for g in self.belief.belief_food:
            d = field[g[0], g[1]]
            if d < 0:
                continue
            if best_d is None or d < best_d:
                best, best_d = tuple(g), d
        if best is not None and best_d <= radius:
            return best
        return None

    def _actions_to(self, src, dst) -> list[int] | None:
        """Grid-BFS action sequence src -> dst (any free cells)."""
        if tuple(src) == tuple(dst):
            return []
        if tuple(dst) in self.env.walls:
            return None
        prev: dict = {}
        dist = bfs_field(
            self.env.walls, self.env.height, self.env.width, tuple(src), prev
        )
        if dist[dst[0], dst[1]] < 0:
            return None
        actions = []
        cur = tuple(dst)
        while cur != tuple(src):
            parent, a = prev[cur]
            actions.append(a)
            cur = parent
        return actions[::-1]

    def _opp_vis(self) -> frozenset:
        if self._last_obs is None:
            return frozenset()
        obs = self._last_obs[self.anchor]
        return frozenset(map(tuple, np.argwhere(obs[:, :, 4] == 1)))

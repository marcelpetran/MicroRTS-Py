"""Scripted team opponents for TeamRoadmapEnv.

Mirrors the C++ CMAPF solver's heuristic DNA: greedy-nearest goal
assignment (sequential, so teammates do not duplicate targets) +
shortest-path following. Uses the env's own BFS (octile, no corner
cutting), so paths match the benchmark's movement rules.

Each member sees the map through the env's team-pooled obs (7 channels),
matching the learning team's "slightly centralized" perception. By default
members additionally start with scenario knowledge (all goal positions,
like the CMAPF solver reading the .scen file); disable per team with
TeamAgent(..., know_all_goals=False).

Personas (for OM pretraining data diversity, ported from the 1v1
SimpleForagingEnv agents to octile movement + 7-channel team obs):
  - greedy  : greedy-nearest believed goal, sequential team assignment
              (the Edelkamp-style default; also the training opponent)
  - simple  : commits to a random believed goal, sticks until gone, no
              team coordination (SimpleAgent DNA)
  - switch  : greedy-nearest, but yields a goal when a visible hostile is
              closer to it (GreedySwitchAgent DNA)
  - stalker : contests winnable goals, loiters adjacent to deny,
              frontier-explorates otherwise (StalkerAgent DNA)

The team-level interface is what QLearningAgent expects from the hostile
side: reset(), select_actions(obs) -> {agent_id: action}, and
get_team_heatmap() -> (H, W) true-intent claim map for eval metrics.

A per-instance BFS distance-field cache is shared across members (fields
depend only on the static walls, so entries never go stale on one map).
"""

import numpy as np

from omexplore.envs.roadmap_foraging_env import MOVES as _MOVES

_FIELD_CACHE_CAP = 2048


class TeamMember:
    """Base: food belief from team-pooled obs + cached BFS distances.

    know_all_goals: seed belief_food with the full goal set at reset()
    (scenario knowledge, like the CMAPF solver reading the .scen file).
    The belief is then corrected by team-pooled obs: believed goals are
    discarded once the team sees their cell empty (e.g. collected by an
    opponent outside vision stays believed until disproven — a rational
    agent with an initial map prior). reset() must be called after
    env.reset(), when the layout is sampled.
    """

    def __init__(self, agent_id, env, field_cache=None, know_all_goals=True):
        self.agent_id = agent_id
        self.env = env
        self.know_all_goals = know_all_goals
        self._fc = field_cache if field_cache is not None else {}
        self.reset()

    def reset(self):
        self.belief_food = (
            set(self.env.food_positions) if self.know_all_goals else set()
        )
        self.belief_opps = set()
        self.target = None
        self.path = []
        self._prev_vis = None

    def update_belief(self, obs):
        """obs: this agent's (team-pooled) 7-channel observation.

        Equivalent to scanning every visible cell each step (food->add,
        else->discard) but only re-verifies believed cells that are
        currently visible and processes newly (re-)visible cells.
        """
        vis = obs[:, :, 6].astype(bool)
        food = obs[:, :, 1].astype(bool)
        for p in list(self.belief_food):
            if vis[p[0], p[1]] and not food[p[0], p[1]]:
                self.belief_food.discard(p)
        if self._prev_vis is None:
            new_vis = vis
        else:
            new_vis = vis & ~self._prev_vis
        fr, fc = np.nonzero(new_vis & food)
        for r, c in zip(fr, fc):
            self.belief_food.add((int(r), int(c)))
        self._prev_vis = vis.copy()

    def update_opp_belief(self, obs):
        """Track last-seen hostile cells from the OPP channel (ch 4)."""
        opp = obs[:, :, 4].astype(bool)
        vis = obs[:, :, 6].astype(bool)
        orows, ocols = np.nonzero(opp)
        for r, c in zip(orows, ocols):
            self.belief_opps.add((int(r), int(c)))
        for p in list(self.belief_opps):
            if vis[p[0], p[1]] and not opp[p[0], p[1]]:
                self.belief_opps.discard(p)

    def _field(self, pos):
        """Cached BFS distance field from pos (walls are static)."""
        f = self._fc.get(pos)
        if f is None:
            f = self.env.bfs_distance_field(pos)
            if len(self._fc) >= _FIELD_CACHE_CAP:
                self._fc.clear()
            self._fc[pos] = f
        return f

    def _dist(self, field, p):
        v = field[p[0], p[1]]
        return v if v >= 0 else np.inf

    def _follow_path(self, my_pos):
        """Pop the next action toward self.target (replan if needed)."""
        if not self.path:
            self.path = self.env.find_path(my_pos, self.target)
            if not self.path:
                # Unreachable or already standing on it (someone else took
                # the goal): forget it and wander this step.
                self.belief_food.discard(self.target)
                self.target = None
                return np.random.randint(8)
        return self.path.pop(0)

    def act(self, taken):
        raise NotImplementedError


class GreedyMember(TeamMember):
    """One scripted agent: greedy-nearest believed goal, BFS path following."""

    def _choose_target(self, taken):
        my_pos = self.env.agents[self.agent_id]
        candidates = [f for f in self.belief_food if f not in taken]
        if not candidates:
            self.target = None
            self.path = []
            return
        field = self._field(my_pos)
        self.target = min(
            candidates,
            key=lambda f: self._dist(field, f),
        )
        self.path = []

    def act(self, taken):
        # Retarget when the current one is gone or claimed by a teammate
        # (processed earlier in the sequential assignment).
        if (
            self.target is None
            or self.target not in self.belief_food
            or self.target in taken
        ):
            self._choose_target(taken)
        if self.target is None:
            return np.random.randint(8)  # wander (explore)
        return self._follow_path(self.env.agents[self.agent_id])


class SimpleMember(TeamMember):
    """Commits to a random believed goal; ignores team coordination."""

    def act(self, taken):
        if self.target is not None and self.target not in self.belief_food:
            self.target = None
            self.path = []
        if self.target is None:
            candidates = list(self.belief_food)
            if not candidates:
                return np.random.randint(8)
            self.target = candidates[np.random.randint(len(candidates))]
            self.path = []
        return self._follow_path(self.env.agents[self.agent_id])


class GreedySwitchMember(TeamMember):
    """Greedy-nearest, but yields goals a visible hostile is closer to."""

    def update_belief(self, obs):
        super().update_belief(obs)
        self.update_opp_belief(obs)

    def act(self, taken):
        my_pos = self.env.agents[self.agent_id]
        my_field = self._field(my_pos)
        cands = [f for f in self.belief_food if f not in taken]
        if not cands:
            self.target = None
            self.path = []
            return np.random.randint(8)
        opp_fields = [self._field(op) for op in self.belief_opps]

        def oppd(p):
            return min((self._dist(f, p) for f in opp_fields), default=np.inf)

        target = (
            self.target
            if self.target in cands
            else min(cands, key=lambda f: self._dist(my_field, f))
        )
        # Abandon the goal if a visible hostile will get there first.
        if opp_fields and oppd(target) < self._dist(my_field, target):
            safer = [f for f in cands if self._dist(my_field, f) <= oppd(f)]
            if safer:
                target = min(safer, key=lambda f: self._dist(my_field, f))
        if target != self.target:
            self.target = target
            self.path = []
        return self._follow_path(my_pos)


class StalkerMember(TeamMember):
    """Hyper-reactive interceptor: contests winnable goals, loiters to deny.

    With hostiles in sight it races to the winnable goal (my_dist <= opp
    dist) closest to the hostile — intercepting rather than collecting —
    and loiters one cell away while the hostile is far. Otherwise it
    greedy-collects or frontier-explores toward the nearest unseen cell.
    """

    def update_belief(self, obs):
        super().update_belief(obs)
        self.update_opp_belief(obs)

    def _loiter(self, my_pos):
        """No-op move: steer into a wall (invalid actions are no-ops)."""
        for dr, dc, a in _MOVES:
            if (my_pos[0] + dr, my_pos[1] + dc) in self.env.walls:
                return a
        return np.random.randint(8)

    @staticmethod
    def _reconstruct(prev, goal):
        if goal not in prev:
            return []
        actions = []
        cur = goal
        while prev[cur] is not None:
            parent, a = prev[cur]
            actions.append(a)
            cur = parent
        return actions[::-1]

    def _explore(self, dist, prev, obs):
        """First step toward the nearest reachable unseen free cell."""
        if obs is None:
            return np.random.randint(8)
        unseen = (obs[:, :, 6] == 0) & (obs[:, :, 5] == 0)
        best_d, best_p = None, None
        for r, c in np.argwhere(unseen):
            d = dist[r, c]
            if d >= 0 and (best_d is None or d < best_d):
                best_d, best_p = d, (int(r), int(c))
        if best_p is None:
            return np.random.randint(8)
        path = self._reconstruct(prev, best_p)
        return path[0] if path else np.random.randint(8)

    def _last_obs(self):
        # obs are team-pooled; any member's last frame carries the same
        # fog mask. Fall back to the member dict on the team agent.
        return getattr(self, "last_obs", None)

    def act(self, taken):
        my_pos = self.env.agents[self.agent_id]
        # One BFS for both my distances and path reconstruction.
        dist, prev = self.env._bfs(my_pos)

        def myd(p):
            return dist[p[0], p[1]] if dist[p[0], p[1]] >= 0 else np.inf

        foods = list(self.belief_food)
        if self.belief_opps and foods:
            opp_fields = [self._field(op) for op in self.belief_opps]

            def oppd(p):
                return min((self._dist(f, p) for f in opp_fields), default=np.inf)

            winnable = [f for f in foods if myd(f) <= oppd(f)]
            if winnable:
                # Contest the goal closest to the hostile (intercept).
                target = min(winnable, key=oppd)
                self.target = target
                self.path = []
                if myd(target) == 1 and oppd(target) > 2:
                    return self._loiter(my_pos)  # deny and wait
            else:
                target = min(foods, key=myd)
                self.target = target
                self.path = []
        elif foods:
            self.target = min(foods, key=myd)
            self.path = []
        else:
            self.target = None
            self.path = []
            return self._explore(dist, prev, self._last_obs())

        path = self._reconstruct(prev, self.target) if self.target else []
        if path:
            return path[0]
        return np.random.randint(8)


PERSONAS = {
    "greedy": GreedyMember,
    "simple": SimpleMember,
    "switch": GreedySwitchMember,
    "stalker": StalkerMember,
}


class TeamAgent:
    """Controls every member of one team with scripted policies.

    personas: tuple of persona names, one per member (default: all greedy).
    know_all_goals: members start knowing every goal position (scenario
    knowledge, CMAPF-solver style); default True.
    """

    def __init__(
        self, env, team_id=1, personas=None, field_cache=None, know_all_goals=True
    ):
        self.env = env
        self.team_id = team_id
        member_ids = env.get_team_members(team_id)
        if personas is None:
            personas = ("greedy",) * len(member_ids)
        if len(personas) != len(member_ids):
            raise ValueError(
                f"personas {personas} must match team size {len(member_ids)}"
            )
        self.personas = tuple(personas)
        self.members = [
            PERSONAS[p](a, env, field_cache=field_cache, know_all_goals=know_all_goals)
            for a, p in zip(member_ids, personas)
        ]

    def reset(self):
        for m in self.members:
            m.reset()

    def select_actions(self, obs):
        for m in self.members:
            m.update_belief(obs[m.agent_id])
            m.last_obs = obs[m.agent_id]
        actions = {}
        taken = set()
        # Sequential greedy assignment: members processed in team order keep
        # their targets; earlier picks exclude goals for later picks. (A
        # cheaper approximation of the C++ solver's centralized assignment.)
        for m in self.members:
            actions[m.agent_id] = m.act(taken)
            if m.target is not None:
                taken.add(m.target)
        return actions

    def get_team_heatmap(self):
        """True-intent claim map: 1.0 at each member's current target."""
        h = np.zeros((self.env.height, self.env.width), dtype=np.float32)
        for m in self.members:
            if m.target is not None:
                h[m.target[0], m.target[1]] += 1.0
        return h

    def get_team_heatmap_sparse(self, exclude_id=None):
        """Sparse intent map [(r, c, weight), ...], excluding one member."""
        counts = {}
        for m in self.members:
            if m.agent_id == exclude_id or m.target is None:
                continue
            counts[m.target] = counts.get(m.target, 0.0) + 1.0
        return [(p[0], p[1], w) for p, w in counts.items()]

"""Scripted team opponents for TeamRoadmapEnv.

Mirrors the C++ CMAPF solver's heuristic DNA: greedy-nearest goal
assignment (sequential, so teammates do not duplicate targets) +
shortest-path following. Uses the env's own BFS (octile, no corner
cutting), so paths match the benchmark's movement rules.

Each member sees the map through the env's team-pooled obs (7 channels),
matching the learning team's "slightly centralized" perception.

The team-level interface is what QLearningAgent expects from the hostile
side: reset(), select_actions(obs) -> {agent_id: action}, and
get_team_heatmap() -> (H, W) true-intent claim map for eval metrics.
"""

import numpy as np


class GreedyMember:
    """One scripted agent: greedy-nearest believed goal, BFS path following."""

    def __init__(self, agent_id, env):
        self.agent_id = agent_id
        self.env = env
        self.reset()

    def reset(self):
        self.belief_food = set()
        self.target = None
        self.path = []

    def update_belief(self, obs):
        """obs: this agent's (team-pooled) 7-channel observation."""
        vis = obs[:, :, 6]
        food = obs[:, :, 1]
        for r, c in np.argwhere(vis == 1):
            if food[r, c] == 1:
                self.belief_food.add((int(r), int(c)))
            else:
                self.belief_food.discard((int(r), int(c)))

    def _choose_target(self, taken):
        my_pos = self.env.agents[self.agent_id]
        candidates = [f for f in self.belief_food if f not in taken]
        if not candidates:
            self.target = None
            self.path = []
            return
        field = self.env.bfs_distance_field(my_pos)
        self.target = min(
            candidates,
            key=lambda f: field[f[0], f[1]] if field[f[0], f[1]] >= 0 else np.inf,
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

        if not self.path:
            self.path = self.env.find_path(self.env.agents[self.agent_id], self.target)
            if not self.path:
                # Unreachable or already standing on it (someone else took
                # the goal): forget it and wander this step.
                self.belief_food.discard(self.target)
                self.target = None
                return np.random.randint(8)
        return self.path.pop(0)


class TeamAgent:
    """Controls every member of one team with greedy scripted policies."""

    def __init__(self, env, team_id=1):
        self.env = env
        self.team_id = team_id
        self.members = [GreedyMember(a, env) for a in env.get_team_members(team_id)]

    def reset(self):
        for m in self.members:
            m.reset()

    def select_actions(self, obs):
        for m in self.members:
            m.update_belief(obs[m.agent_id])
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

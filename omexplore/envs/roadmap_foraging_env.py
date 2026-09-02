"""Scaled foraging environment on MovingAI benchmark maps (CMAPF).

Drop-in replacement for SimpleForagingEnv with an identical public
interface (reset / step / action_space / get_global_state / swap_agents /
reset_random_spawn / get_visibility_map / render), so QNet,
SpatialOpponentModel and BeliefTracker plug in unchanged.

Differences vs SimpleForagingEnv, forced by scale:

- Map + scenario come from cmapf_unified/maps/<name>.map and .scen.
  Agents spawn at scenario starts, food at scenario goals.
- Movement is octile (8 directions), matching the MovingAI benchmark and
  the C++ solvers: actions 0-3 Up/Down/Left/Right, 4-7 the diagonals
  (UpLeft, UpRight, DownLeft, DownRight). Corner cutting is disallowed —
  a diagonal move requires BOTH orthogonal neighbors to be free — which
  reproduces the scen files' optimal distances exactly (verified to 1e-4).
- Visibility is computed lazily per visited cell and cached. The original
  O(H^2 W^2) precompute is fine at 11x11 but explodes at 81x65.
- precompute_paths() (all-pairs A*) is replaced by bfs_distance_field()
  and find_path(); an all-pairs table is infeasible at this scale.

Remember to update OMGArgs.state_shape to (H, W, 6) for the chosen map,
e.g. (81, 65, 6) for den312d, and scale max_steps (~400) and gamma (~0.995).
The Q agents pick up the action space automatically via
len(env.action_space) == 8.

TeamRoadmapEnv (same file) extends this with team-based competitive
exploration: several teams with pooled team vision, shared team
rewards, a 7-channel obs (self / teammates / opponents) and per-team
coverage metrics — the setting for the teammate/opponent-modeling
experiments. Set OMGArgs.state_shape to (H, W, 7) for it.
"""

import os
from collections import deque

import numpy as np

MAPS_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "cmapf_unified", "maps"
    )
)

# 0: Up, 1: Down, 2: Left, 3: Right,
# 4: UpLeft, 5: UpRight, 6: DownLeft, 7: DownRight
MOVES = [
    (-1, 0, 0),
    (1, 0, 1),
    (0, -1, 2),
    (0, 1, 3),
    (-1, -1, 4),
    (-1, 1, 5),
    (1, -1, 6),
    (1, 1, 7),
]
PASSABLE_CHARS = {".", "G"}


def load_movingai_map(path: str) -> list[str]:
    """Parse a MovingAI .map into normalized rows of '.' (free) / '#' (wall)."""
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]
    # Header: "type octile", "height H", "width W", "map"
    height = int(next(l for l in lines if l.startswith("height")).split()[1])
    width = int(next(l for l in lines if l.startswith("width")).split()[1])
    grid = lines[4 : 4 + height]
    assert len(grid) == height, f"map rows {len(grid)} != header height {height}"
    rows = []
    for row in grid:
        assert len(row) == width, f"row width {len(row)} != header width {width}"
        rows.append("".join("." if ch in PASSABLE_CHARS else "#" for ch in row))
    return rows


def load_scen(path: str) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    """Parse a MovingAI .scen into [((row, col) start, (row, col) goal, dist), ...].

    Scen columns: bucket map w h startx starty goalx goaly dist.
    x is the column, y is the row, so position = (y, x) in (r, c) order.
    """
    pairs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip() or line.startswith("version"):
                continue
            parts = line.split()
            assert len(parts) >= 9, f"malformed scen line {i}: {line!r}"
            sx, sy, gx, gy, dist = (
                int(parts[4]),
                int(parts[5]),
                int(parts[6]),
                int(parts[7]),
                float(parts[8]),
            )
            pairs.append(((sy, sx), (gy, gx), dist))
    return pairs


class RoadmapForagingEnv:
    """Competitive foraging on a large MovingAI grid map with fog of war.

    Semantics deliberately mirror SimpleForagingEnv except movement:
    - 6 int8 channels: 0 empty, 1 food, 2 self, 3 opponent, 4 wall, 5 vis mask
    - actions 0-7 move one cell in 8 directions (octile); diagonals require
      both orthogonal neighbors free (no corner cutting); anything else is
      a no-op
    - agents may share a cell; if both stand on food they split 0.5/0.5
    - terminal when food is exhausted or max_steps is reached
    """

    def __init__(
        self,
        map_name: str = "den312d",
        max_steps: int = 400,
        vision_radius: int = 5,
        num_food: int = 8,
        scen_offset: int = 0,
        maps_dir: str | None = None,
    ):
        maps_dir = maps_dir or MAPS_DIR
        self.map_layout: list[str] = load_movingai_map(
            os.path.join(maps_dir, f"{map_name}.map")
        )
        self.scen: list = load_scen(os.path.join(maps_dir, f"{map_name}-even-1.scen"))
        self.height: int = len(self.map_layout)
        self.width: int = len(self.map_layout[0])
        self.max_steps: int = max_steps
        self.vision_radius: int = vision_radius
        self.action_space: list[int] = self._get_action_space()

        self.walls = {
            (r, c)
            for r, row in enumerate(self.map_layout)
            for c, ch in enumerate(row)
            if ch == "#"
        }

        # Layout hook: TeamRoadmapEnv overrides _init_layout() to change
        # agent count, channel count and spawn/goal logic.
        self._num_food_target = num_food
        self.scen_offset = scen_offset
        self._init_layout()

        # Vectorized wall coordinates for observation construction.
        if self.walls:
            w_rows, w_cols = zip(*self.walls)
            self._wall_rows = np.asarray(w_rows, dtype=np.int64)
            self._wall_cols = np.asarray(w_cols, dtype=np.int64)
        else:
            self._wall_rows = np.empty(0, dtype=np.int64)
            self._wall_cols = np.empty(0, dtype=np.int64)

        self.base_obs = np.zeros(
            (self.height, self.width, self.features), dtype=np.int8
        )
        self.base_obs[:, :, 0] = 1
        self.base_obs[self._wall_rows, self._wall_cols, 0] = 0
        self.base_obs[self._wall_rows, self._wall_cols, 4] = 1

        # Lazy visibility cache: position -> HxW int8 mask. Keyed lookups are
        # cheap; only cells actually visited are ever computed (~121 LOS
        # checks each). Cleared wholesale if it grows too large.
        self._vis_cache: dict = {}
        self._vis_cache_max = 4096

        self.reset()

    def _init_layout(self):
        """Set num_agents, features, and the fixed initial layout."""
        self.num_agents: int = 2
        self.features: int = 6
        rows = [
            row
            for row in self.scen
            if row[0] not in self.walls and row[1] not in self.walls
        ]
        off = self.scen_offset % max(1, len(rows) - self._num_food_target - 2)
        self._scen_rows = rows
        self._initial_agents = {0: rows[off][0], 1: rows[off + 1][0]}
        self._initial_food = {
            rows[off + 2 + i][1] for i in range(self._num_food_target)
        }
        self.num_food = len(self._initial_food)

    def _try_move(self, agent_id, action):
        """Resolve one action to a legal new position; illegal -> no-op."""
        r, c = self.agents[agent_id]

        if action == 0:
            dr, dc = -1, 0
        elif action == 1:
            dr, dc = 1, 0
        elif action == 2:
            dr, dc = 0, -1
        elif action == 3:
            dr, dc = 0, 1
        elif action == 4:
            dr, dc = -1, -1
        elif action == 5:
            dr, dc = -1, 1
        elif action == 6:
            dr, dc = 1, -1
        elif action == 7:
            dr, dc = 1, 1
        else:
            dr = dc = 0

        nr, nc = r + dr, c + dc
        legal = 0 <= nr < self.height and 0 <= nc < self.width
        if legal and (nr, nc) in self.walls:
            legal = False
        if legal and dr and dc:
            # No corner cutting: both orthogonal neighbors must be free.
            if (r + dr, c) in self.walls or (r, c + dc) in self.walls:
                legal = False
        return (nr, nc) if legal else self.agents[agent_id]

    # ------------------------------------------------------------------ #
    # Core API (identical semantics to SimpleForagingEnv)                 #
    # ------------------------------------------------------------------ #

    def reset(self):
        self.agents = self._initial_agents.copy()
        self.food_positions = self._initial_food.copy()
        self.steps = 0
        self.rewards = {0: 0, 1: 0}
        self.terminal = False
        return self._get_ego_centric_obs()

    def step(self, actions):
        rewards = {0: 0.0, 1: 0.0}
        new_positions = {}

        for agent_id, action in actions.items():
            new_positions[agent_id] = self._try_move(agent_id, action)

        self.agents = new_positions
        self.steps += 1

        pos0 = self.agents[0]
        pos1 = self.agents[1]

        if pos0 == pos1 and pos0 in self.food_positions:
            rewards[0] += 0.5
            rewards[1] += 0.5
            self.food_positions.remove(pos0)
        else:
            if pos0 in self.food_positions:
                rewards[0] += 1.0
                self.food_positions.remove(pos0)
            if pos1 in self.food_positions:
                rewards[1] += 1.0
                self.food_positions.remove(pos1)

        return self._get_ego_centric_obs(), rewards, self._check_terminal(), {}

    def swap_agents(self):
        self.agents[0] = self._initial_agents[1]
        self.agents[1] = self._initial_agents[0]
        return self._get_ego_centric_obs()

    def reset_random_spawn(self):
        _ = self.reset()

        # Remove a random food
        if np.random.rand() > 0.5:
            food_list = list(self.food_positions)
            if len(food_list) > 0:
                removed_food = food_list[np.random.randint(len(food_list))]
                self.food_positions.remove(removed_food)

        freed = self._get_freed_positions()
        A_pos = freed[np.random.randint(0, len(freed))]
        B_pos = freed[np.random.randint(0, len(freed))]
        while B_pos == A_pos:  # avoid identical spawns on a big map
            B_pos = freed[np.random.randint(0, len(freed))]
        self.agents[0] = A_pos
        self.agents[1] = B_pos
        return self._get_ego_centric_obs()

    def get_global_state(self):
        obs = np.zeros((self.height, self.width, self.features), dtype=np.int8)
        obs[:, :, 5] = 1  # All cells visible in global state

        if self._wall_rows.size:
            obs[self._wall_rows, self._wall_cols, 4] = 1

        if self.food_positions:
            food_rows, food_cols = zip(*self.food_positions)
            obs[food_rows, food_cols, 1] = 1

        if self.agents[0] is not None:
            obs[self.agents[0][0], self.agents[0][1], 2] = 1
        if self.agents[1] is not None:
            obs[self.agents[1][0], self.agents[1][1], 3] = 1

        occupied = (obs[..., 1] | obs[..., 2] | obs[..., 3] | obs[..., 4]).astype(bool)
        obs[..., 0] = ~occupied
        return obs

    def get_visibility_map(self, agent_id):
        pos = self.agents[agent_id]
        if pos is None:
            return np.zeros((self.height, self.width), dtype=np.int8)
        cached = self._vis_cache.get(pos)
        if cached is not None:
            return cached.copy()
        if len(self._vis_cache) >= self._vis_cache_max:
            self._vis_cache.clear()
        vis_map = self._compute_visibility(pos)
        self._vis_cache[pos] = vis_map
        return vis_map.copy()

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _get_action_space(self):
        return [0, 1, 2, 3, 4, 5, 6, 7]

    def _get_freed_positions(self):
        occupied = self.food_positions.union(self.walls)
        return [
            (i, j)
            for i in range(self.height)
            for j in range(self.width)
            if (i, j) not in occupied
        ]

    def _compute_visibility(self, agent_pos):
        """Chebyshev-radius + line-of-sight mask, same rule as the small env.

        Only the (2r+1)^2 window around the agent needs LOS checks, so cost
        is independent of map size.
        """
        r_agent, c_agent = agent_pos
        vis_map = np.zeros((self.height, self.width), dtype=np.int8)
        r_lo = max(0, r_agent - self.vision_radius)
        r_hi = min(self.height - 1, r_agent + self.vision_radius)
        c_lo = max(0, c_agent - self.vision_radius)
        c_hi = min(self.width - 1, c_agent + self.vision_radius)
        for r in range(r_lo, r_hi + 1):
            for c in range(c_lo, c_hi + 1):
                if self._has_line_of_sight(r_agent, c_agent, r, c):
                    vis_map[r, c] = 1
        return vis_map

    def _has_line_of_sight(self, r0, c0, r1, c1):
        if (r0, c0) == (r1, c1):
            return True
        steps = max(abs(r1 - r0), abs(c1 - c0)) * 5
        for i in range(1, steps):
            t = i / steps
            r = r0 + 0.5 + t * (r1 - r0)
            c = c0 + 0.5 + t * (c1 - c0)
            tile_r, tile_c = int(np.floor(r)), int(np.floor(c))
            if (tile_r, tile_c) == (r1, c1):
                return True
            if (tile_r, tile_c) != (r0, c0) and (tile_r, tile_c) in self.walls:
                return False
        return True

    def _get_observations(self):
        wall_rows, wall_cols = self._wall_rows, self._wall_cols

        if self.food_positions:
            food_rows, food_cols = zip(*self.food_positions)
            food_rows = np.asarray(food_rows, dtype=np.int64)
            food_cols = np.asarray(food_cols, dtype=np.int64)
        else:
            food_rows = food_cols = np.empty(0, dtype=np.int64)

        observations = {}
        for agent_id in self.agents:
            obs = np.zeros((self.height, self.width, self.features), dtype=np.int8)
            vis_map = self.get_visibility_map(agent_id)
            obs[:, :, 5] = vis_map

            # Walls (known to all agents regardless of visibility)
            if wall_rows.size:
                obs[wall_rows, wall_cols, 4] = 1

            # Visible food
            if food_rows.size:
                visible = vis_map[food_rows, food_cols] == 1
                obs[food_rows[visible], food_cols[visible], 1] = 1

            # Visible agents
            if self.agents[0] is not None:
                r0, c0 = self.agents[0]
                if vis_map[r0, c0] == 1:
                    obs[r0, c0, 2] = 1
            if self.agents[1] is not None:
                r1, c1 = self.agents[1]
                if vis_map[r1, c1] == 1:
                    obs[r1, c1, 3] = 1

            occupied = (obs[..., 1] | obs[..., 2] | obs[..., 3] | obs[..., 4]).astype(
                bool
            )
            obs[..., 0] = vis_map.astype(bool) & ~occupied

            observations[agent_id] = obs
        return observations

    def _get_ego_centric_obs(self):
        obs = self._get_observations()
        obs_0 = obs[0].copy()
        obs_1 = obs[1].copy()
        obs_1[:, :, [2, 3]] = obs_1[:, :, [3, 2]]
        return {0: obs_0, 1: obs_1}

    def _check_terminal(self):
        if self.steps >= self.max_steps or len(self.food_positions) == 0:
            self.terminal = True
        return self.terminal

    # ------------------------------------------------------------------ #
    # Precomputed shortest paths (scaled replacement for precompute_paths)#
    # ------------------------------------------------------------------ #

    def _bfs(self, source):
        """BFS from source over free cells.

        Returns (dist, prev) where dist is an HxW int32 array (-1 unreachable)
        and prev maps cell -> (parent_cell, action) for path reconstruction.
        """
        dist = np.full((self.height, self.width), -1, dtype=np.int32)
        prev: dict = {source: None}
        if source in self.walls:
            return dist, prev
        dist[source] = 0
        q = deque([source])
        while q:
            r, c = q.popleft()
            for dr, dc, a in MOVES:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if (nr, nc) not in self.walls and dist[nr, nc] < 0:
                        if (
                            dr
                            and dc
                            and ((r + dr, c) in self.walls or (r, c + dc) in self.walls)
                        ):
                            continue  # no corner cutting, same rule as step()
                        dist[nr, nc] = dist[r, c] + 1
                        prev[(nr, nc)] = ((r, c), a)
                        q.append((nr, nc))
        return dist, prev

    def bfs_distance_field(self, source) -> np.ndarray:
        """HxW int32 shortest-step distances from source (-1 unreachable).

        The scaled analogue of the food->all-cells path table: one BFS per
        food position gives a distance prior for belief channels and
        subgoal heatmaps in O(H*W).
        """
        return self._bfs(source)[0]

    def find_path(self, start, goal) -> list[int]:
        """BFS shortest path from start to goal as a list of actions 0-7.

        Drop-in replacement for a_star_path(); computing on demand is cheap
        (single BFS over 5k cells) so no all-pairs table is needed.
        """
        if start == goal:
            return []
        dist, prev = self._bfs(start)
        if dist[goal] < 0:
            return []
        actions = []
        cur = goal
        while prev[cur] is not None:
            parent, a = prev[cur]
            actions.append(a)
            cur = parent
        return actions[::-1]

    # ------------------------------------------------------------------ #
    # Rendering (same channel semantics as SimpleForagingEnv)             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def render_from_obs(obs):
        h, w = obs.shape[0], obs.shape[1]
        has_fog = obs.shape[2] >= 6
        render_grid = np.full((h, w), ".", dtype=str)
        for i in range(h):
            for j in range(w):
                if has_fog and obs[i, j, 5] == 0:
                    render_grid[i, j] = "?"
                elif obs[i, j, 4] == 1:
                    render_grid[i, j] = "#"
                elif obs[i, j, 1] == 1:
                    render_grid[i, j] = "F"
                elif obs[i, j, 2] == 1 and obs[i, j, 3] == 1:
                    render_grid[i, j] = "X"
                elif obs[i, j, 2] == 1:
                    render_grid[i, j] = "A"
                elif obs[i, j, 3] == 1:
                    render_grid[i, j] = "B"
        for row in render_grid:
            print(" ".join(row))
        print()

    def render(self, agent_id=0):
        obs = self._get_ego_centric_obs()[agent_id]
        self.render_from_obs(obs)

    def render_global(self):
        obs = self.get_global_state()
        self.render_from_obs(obs)


class TeamRoadmapEnv(RoadmapForagingEnv):
    """Team-based competitive exploration on a MovingAI map with fog of war.

    Several teams compete for a shared pool of hidden goals. Teammates pool
    their vision (slightly centralized perception) and share rewards: when
    any member of a team stands on a goal, EVERY member of that team
    receives the reward. If agents from several teams stand on the same
    goal in the same step, the reward is split evenly between the teams
    (0.5/0.5 for two teams).

    Observations (per agent, egocentric, 7 int8 channels):
      0 empty (team-visible and unoccupied)
      1 goal (visible through the TEAM's pooled vision)
      2 self
      3 teammate (always shown, even outside own vision)
      4 opponent (only when inside the team's pooled vision)
      5 wall (static map knowledge)
      6 team vision union (fog mask)

    Opponent/teammate-modeling hooks (claim-count variant):
      - get_agent_positions() returns the true positions of ALL agents
        (friends and foes; obs itself carries no team labels for foes).
      - step()'s info dict carries "collected" (goal cells) and "collectors"
        (goal -> agent ids standing on it), enough for hindsight per-agent
        subgoal labeling; aggregate per team for claim-count targets.
      - The OM predicts a per-cell expected claim count per team
        (friendly / hostile), so outputs no longer sum to 1 across cells;
        train with Poisson NLL (or Huber as a fallback).
    """

    def __init__(
        self,
        map_name: str = "den312d",
        max_steps: int = 400,
        vision_radius: int = 5,
        num_goals: int = 16,
        team_sizes: tuple = (2, 2),
        maps_dir: str | None = None,
    ):
        self.team_sizes = tuple(team_sizes)
        self.num_teams = len(self.team_sizes)
        self._num_goals_target = num_goals
        super().__init__(
            map_name=map_name,
            max_steps=max_steps,
            vision_radius=vision_radius,
            maps_dir=maps_dir,
        )

    def _init_layout(self):
        self.num_agents = sum(self.team_sizes)
        self.features = 7
        self._num_food_target = self._num_goals_target
        self._scen_rows = [
            row
            for row in self.scen
            if row[0] not in self.walls and row[1] not in self.walls
        ]
        # agent_id -> team_id; agents are assigned consecutively per team.
        self.teams = {}
        self._team_members = {t: [] for t in range(self.num_teams)}
        for t, size in enumerate(self.team_sizes):
            for _ in range(size):
                a = len(self.teams)
                self.teams[a] = t
                self._team_members[t].append(a)
        # Safe defaults; _sample_layout() (called from reset()) fills these.
        self.agents = {}
        self.food_positions = set()
        self.num_food = 0
        self.team_scores = {t: 0.0 for t in range(self.num_teams)}
        self._team_vis = {
            t: np.zeros((self.height, self.width), dtype=np.int8)
            for t in range(self.num_teams)
        }
        self._coverage = {
            t: np.zeros((self.height, self.width), dtype=bool)
            for t in range(self.num_teams)
        }

    def _sample_layout(self):
        """Sample starts (one per agent) and goals from the scen rows."""
        perm = np.random.permutation(len(self._scen_rows))
        used = set()
        starts = []
        for i in perm:
            s = self._scen_rows[i][0]
            if s not in used:
                starts.append(s)
                used.add(s)
            if len(starts) == self.num_agents:
                break
        if len(starts) < self.num_agents:  # tiny-scen fallback
            for p in self._get_freed_positions():
                if p not in used:
                    starts.append(p)
                    used.add(p)
                if len(starts) == self.num_agents:
                    break
        goals = []
        for i in perm:
            g = self._scen_rows[i][1]
            if g not in used:
                goals.append(g)
                used.add(g)
            if len(goals) == self._num_goals_target:
                break
        if len(goals) < self._num_goals_target:
            for p in self._get_freed_positions():
                if p not in used:
                    goals.append(p)
                    used.add(p)
                if len(goals) == self._num_goals_target:
                    break
        self._initial_agents = {a: starts[a] for a in range(self.num_agents)}
        self._initial_food = set(goals)
        self.num_food = len(goals)

    def reset(self):
        self._sample_layout()
        self.agents = self._initial_agents.copy()
        self.food_positions = self._initial_food.copy()
        self.steps = 0
        self.rewards = {a: 0 for a in self.agents}
        self.terminal = False
        self.team_scores = {t: 0.0 for t in range(self.num_teams)}
        self._coverage = {
            t: np.zeros((self.height, self.width), dtype=bool)
            for t in range(self.num_teams)
        }
        self._update_team_vis()
        return self._get_ego_centric_obs()

    def reset_random_spawn(self):
        return self.reset()

    def swap_agents(self):
        """Reset and swap team-0/team-1 spawn positions (training-loop compat)."""
        obs = self.reset()
        if self.num_teams >= 2:
            for a, b in zip(self._team_members[0], self._team_members[1]):
                self.agents[a], self.agents[b] = self.agents[b], self.agents[a]
            self._update_team_vis()
            obs = self._get_ego_centric_obs()
        return obs

    def step(self, actions):
        rewards = {a: 0.0 for a in self.agents}
        new_positions = {}
        for agent_id in self.agents:
            new_positions[agent_id] = self._try_move(agent_id, actions.get(agent_id))
        self.agents = new_positions
        self.steps += 1
        self._update_team_vis()

        collected = []
        collectors = {}
        team_rewards = {}
        for goal in list(self.food_positions):
            on = sorted(a for a, pos in self.agents.items() if pos == goal)
            teams_on = {self.teams[a] for a in on}
            if not teams_on:
                continue
            share = 1.0 / len(teams_on)
            self.food_positions.remove(goal)
            collected.append(goal)
            collectors[goal] = on
            for t in teams_on:
                self.team_scores[t] += share
                team_rewards[t] = team_rewards.get(t, 0.0) + share
                for a in self._team_members[t]:
                    rewards[a] += share

        info = {
            "collected": collected,
            "collectors": collectors,
            "team_rewards": team_rewards,
            "team_scores": dict(self.team_scores),
            "coverage": {t: self.get_coverage(t) for t in range(self.num_teams)},
        }
        return self._get_ego_centric_obs(), rewards, self._check_terminal(), info

    def _update_team_vis(self):
        for t in range(self.num_teams):
            vis = np.zeros((self.height, self.width), dtype=np.int8)
            for a in self._team_members[t]:
                if self.agents.get(a) is not None:
                    vis |= self.get_visibility_map(a)
            self._team_vis[t] = vis
            self._coverage[t] |= vis.astype(bool)

    def _get_observations(self):
        wall_rows, wall_cols = self._wall_rows, self._wall_cols
        food = list(self.food_positions)
        if food:
            food_rows = np.asarray([p[0] for p in food], dtype=np.int64)
            food_cols = np.asarray([p[1] for p in food], dtype=np.int64)
        else:
            food_rows = food_cols = np.empty(0, dtype=np.int64)

        observations = {}
        for agent_id in self.agents:
            t = self.teams[agent_id]
            team_vis = self._team_vis[t]
            obs = np.zeros((self.height, self.width, self.features), dtype=np.int8)
            obs[:, :, 6] = team_vis

            if wall_rows.size:
                obs[wall_rows, wall_cols, 5] = 1

            if food_rows.size:
                visible = team_vis[food_rows, food_cols] == 1
                obs[food_rows[visible], food_cols[visible], 1] = 1

            r_self, c_self = self.agents[agent_id]
            obs[r_self, c_self, 2] = 1
            for mate in self._team_members[t]:
                if mate == agent_id:
                    continue
                pos = self.agents.get(mate)
                if pos is not None:
                    obs[pos[0], pos[1], 3] = 1
            for other, pos in self.agents.items():
                if self.teams[other] != t and pos is not None:
                    if team_vis[pos[0], pos[1]]:
                        obs[pos[0], pos[1], 4] = 1

            occupied = (
                obs[..., 1] | obs[..., 2] | obs[..., 3] | obs[..., 4] | obs[..., 5]
            ).astype(bool)
            obs[..., 0] = team_vis.astype(bool) & ~occupied
            observations[agent_id] = obs
        return observations

    def _get_ego_centric_obs(self):
        # Already self-anchored: channel 2 is always the observing agent.
        return self._get_observations()

    def get_global_state(self):
        """Oracle state: 0 empty, 1 goal, 2+t team t, T+2 wall, T+3 all-vis."""
        T = self.num_teams
        obs = np.zeros((self.height, self.width, T + 4), dtype=np.int8)
        obs[:, :, T + 3] = 1
        if self._wall_rows.size:
            obs[self._wall_rows, self._wall_cols, T + 2] = 1
        if self.food_positions:
            rows, cols = zip(*self.food_positions)
            obs[list(rows), list(cols), 1] = 1
        for a, pos in self.agents.items():
            if pos is not None:
                obs[pos[0], pos[1], 2 + self.teams[a]] = 1
        obs[..., 0] = ~obs[..., 1 : T + 3].any(axis=2)
        return obs

    # ------------------------------------------------------------------ #
    # Introspection / OM labeling hooks                                   #
    # ------------------------------------------------------------------ #

    def get_agent_positions(self):
        """True positions of ALL agents (oracle for hindsight labeling)."""
        return dict(self.agents)

    def get_team_members(self, team_id):
        return list(self._team_members[team_id])

    def get_coverage(self, team_id):
        """Fraction of passable cells ever seen by team_id this episode."""
        free = self.height * self.width - len(self.walls)
        return float(self._coverage[team_id].sum()) / max(free, 1)

    # ------------------------------------------------------------------ #
    # Rendering                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def render_from_obs(obs):
        h, w = obs.shape[0], obs.shape[1]
        has_fog = obs.shape[2] >= 7
        grid = np.full((h, w), ".", dtype=str)
        for i in range(h):
            for j in range(w):
                if has_fog and obs[i, j, 6] == 0:
                    grid[i, j] = "?"
                elif obs[i, j, 5] == 1:
                    grid[i, j] = "#"
                elif obs[i, j, 1] == 1:
                    grid[i, j] = "F"
                elif obs[i, j, 2] == 1:
                    grid[i, j] = "A"
                elif obs[i, j, 3] == 1:
                    grid[i, j] = "a"
                elif obs[i, j, 4] == 1:
                    grid[i, j] = "B"
        for row in grid:
            print(" ".join(row))
        print()

    def render_global(self):
        grid = np.full((self.height, self.width), ".", dtype=str)
        for r, c in self.walls:
            grid[r, c] = "#"
        for r, c in self.food_positions:
            grid[r, c] = "F"
        for a, pos in self.agents.items():
            if pos is not None:
                grid[pos[0], pos[1]] = str(self.teams[a])
        for row in grid:
            print(" ".join(row))
        print()


if __name__ == "__main__":
    import time

    env = RoadmapForagingEnv()
    print(
        f"map den312d: {env.height}x{env.width}, "
        f"{len(env.walls)} walls, {len(env.scen)} scen pairs, "
        f"{env.num_food} food, vision r={env.vision_radius}"
    )

    # --- Sanity 1: scen coordinates must land on passable cells.
    bad = sum(1 for s, g, _ in env._scen_rows if s in env.walls or g in env.walls)
    print(f"scen rows on walls: {bad} (expect 0; >0 means x/y are swapped)")

    # --- Sanity 2: octile Dijkstra (same movement rules as step()) must
    # reproduce the scen's recorded optimal distances exactly.
    import heapq
    import math

    SQRT2 = math.sqrt(2)

    def octile_dijkstra(src):
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, cur = heapq.heappop(pq)
            if d > dist.get(cur, 1e18):
                continue
            r, c = cur
            for dr, dc, _ in MOVES:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < env.height and 0 <= nc < env.width):
                    continue
                if (nr, nc) in env.walls:
                    continue
                if dr and dc and ((r + dr, c) in env.walls or (r, c + dc) in env.walls):
                    continue
                nd = d + (SQRT2 if dr and dc else 1.0)
                if nd < dist.get((nr, nc), 1e18) - 1e-12:
                    dist[(nr, nc)] = nd
                    heapq.heappush(pq, (nd, (nr, nc)))
        return dist

    bad = 0
    for s, g, d in env._scen_rows[:20]:
        got = octile_dijkstra(s).get(g, -1)
        if abs(got - d) > 1e-4:
            bad += 1
            print(f"  dist mismatch {s}->{g}: dijkstra={got} scen={d}")
    print(f"octile distance mismatches in first 20 scen rows: {bad} (expect 0)")

    # --- Smoke test: random-policy episode.
    obs = env.reset()
    print(f"obs shape: {obs[0].shape} (set OMGArgs.state_shape to this)")
    t0 = time.time()
    steps = 0
    ret = {0: 0.0, 1: 0.0}
    done = False
    while not done:
        actions = {0: np.random.randint(8), 1: np.random.randint(8)}
        obs, rewards, done, _ = env.step(actions)
        ret[0] += rewards[0]
        ret[1] += rewards[1]
        steps += 1
    dt = time.time() - t0
    print(
        f"episode: {steps} steps in {dt:.2f}s "
        f"({1000 * dt / max(steps, 1):.2f} ms/step), "
        f"returns A={ret[0]:.1f} B={ret[1]:.1f}, "
        f"food left={len(env.food_positions)}"
    )

    # --- Smoke test: path finding (verify by simulating the path).
    s, g, d = env._scen_rows[0]
    path = env.find_path(s, g)
    r, c = s
    for a in path:
        dr, dc = next((dr, dc) for dr, dc, aa in MOVES if aa == a)
        r, c = r + dr, c + dc
    arrived = (r, c) == g
    steps_to_goal = env.bfs_distance_field(s)[g]
    print(
        f"find_path {s}->{g}: {len(path)} actions, arrives={arrived}, "
        f"bfs steps={steps_to_goal}, scen octile optimum={d}"
    )

    # --- Render agent 0's (fogged) view and the global state once.
    print("\nAgent 0 view ('?' = fog):")
    env.render(0)

    # ================================================================== #
    # TeamRoadmapEnv smoke tests                                         #
    # ================================================================== #
    print("\n=== TeamRoadmapEnv (2v2 competitive exploration) ===")
    tenv = TeamRoadmapEnv(num_goals=16, team_sizes=(2, 2), max_steps=400)
    tobs = tenv.reset()
    assert tobs[0].shape == (tenv.height, tenv.width, 7), tobs[0].shape
    assert tenv.get_global_state().shape == (
        tenv.height,
        tenv.width,
        tenv.num_teams + 4,
    )
    print(
        f"teams: {tenv.team_sizes}, agents: {tenv.num_agents}, "
        f"goals: {tenv.num_food}, obs {tobs[0].shape}, "
        f"global {tenv.get_global_state().shape}"
    )

    # --- Smoke test: random-policy episode; team-shared rewards must give
    # every member exactly the team score (identical float additions).
    t0 = time.time()
    steps = 0
    ret = {a: 0.0 for a in range(tenv.num_agents)}
    done = False
    while not done:
        actions = {a: np.random.randint(8) for a in range(tenv.num_agents)}
        tobs, rewards, done, info = tenv.step(actions)
        for a, r in rewards.items():
            ret[a] += r
        steps += 1
    dt = time.time() - t0
    for t in range(tenv.num_teams):
        for a in tenv.get_team_members(t):
            assert ret[a] == tenv.team_scores[t], (ret, tenv.team_scores)
    cov = {t: round(tenv.get_coverage(t), 3) for t in range(tenv.num_teams)}
    print(
        f"episode: {steps} steps in {dt:.2f}s "
        f"({1000 * dt / max(steps, 1):.2f} ms/step), returns {ret}, "
        f"team scores {dict(tenv.team_scores)}, coverage {cov}, "
        f"goals left={len(tenv.food_positions)}"
    )

    # --- Targeted (a): only agent 0 on a goal -> whole team 0 gets 1.0.
    tenv.reset()
    goal = next(iter(tenv.food_positions))
    freed = [p for p in tenv._get_freed_positions() if p != goal]
    tenv.agents = {0: goal, 1: freed[0], 2: freed[1], 3: freed[2]}
    tenv._update_team_vis()
    _, rewards, _, info = tenv.step({a: None for a in range(4)})
    assert rewards == {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0}, rewards
    assert info["team_rewards"] == {0: 1.0}, info["team_rewards"]
    print("team-shared reward: OK (agent 0 on goal -> both team-0 agents +1.0)")

    # --- Targeted (b): agents 0 and 2 on the same goal -> 0.5 per team.
    tenv.reset()
    goal = next(iter(tenv.food_positions))
    freed = [p for p in tenv._get_freed_positions() if p != goal]
    tenv.agents = {0: goal, 1: freed[0], 2: goal, 3: freed[1]}
    tenv._update_team_vis()
    _, rewards, _, info = tenv.step({a: None for a in range(4)})
    assert rewards == {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}, rewards
    assert info["team_rewards"] == {0: 0.5, 1: 0.5}, info["team_rewards"]
    print("cross-team tie: OK (split 0.5/0.5 between the two teams)")

    # --- Targeted (c): goal visible ONLY through the teammate's vision.
    tenv.reset()
    goal = next(iter(tenv.food_positions))
    gr, gc = goal
    mate = None
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        p = (gr + dr, gc + dc)
        if 0 <= p[0] < tenv.height and 0 <= p[1] < tenv.width and p not in tenv.walls:
            mate = p
            break
    assert mate is not None, "goal has no free orthogonal neighbor"
    far = [
        p
        for p in tenv._get_freed_positions()
        if max(abs(p[0] - gr), abs(p[1] - gc)) > tenv.vision_radius + 2
        and max(abs(p[0] - mate[0]), abs(p[1] - mate[1])) > tenv.vision_radius + 2
    ]
    assert len(far) >= 3, "not enough far cells for the vision test"
    tenv.agents = {0: far[0], 1: mate, 2: far[1], 3: far[2]}
    tenv._update_team_vis()
    own = tenv.get_visibility_map(0)
    assert own[gr, gc] == 0, "goal should be outside self's own vision"
    tobs = tenv._get_ego_centric_obs()
    assert tobs[0][gr, gc, 1] == 1, "goal must be visible via teammate vision"
    assert tobs[0][gr, gc, 6] == 1
    assert tobs[0][mate[0], mate[1], 3] == 1, "teammate always shown"
    print("shared team vision: OK (goal visible only through the teammate)")

    # --- swap_agents() API compat.
    tenv.swap_agents()
    print("swap_agents: OK")

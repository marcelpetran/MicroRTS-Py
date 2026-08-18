import heapq

import numpy as np

from maps import MAP_1


class SimpleForagingEnv:
    def __init__(
        self, max_steps: int = 50, map_layout: list[str] = MAP_1, vision_radius: int = 2
    ):
        self.map_layout: list[str] = map_layout
        self.height: int = len(map_layout)
        self.width: int = len(map_layout[0])
        self.num_agents: int = 2
        # 0: empty, 1: food, 2: agent1, 3: agent2, 4: wall, 5: vis_mask
        self.features: int = 6
        self.max_steps: int = max_steps
        self.vision_radius: int = vision_radius
        self.action_space: list[int] = self._get_action_space()

        self._initial_agents = {0: None, 1: None}
        self._initial_food = set()
        self.walls = set()

        for i, row in enumerate(self.map_layout):
            for j, char in enumerate(row):
                pos = (i, j)
                if char == "#":
                    self.walls.add(pos)
                elif char == "o":
                    self._initial_food.add(pos)
                elif char == "A":
                    self._initial_agents[0] = pos
                elif char == "B":
                    self._initial_agents[1] = pos

        self.num_food = len(self._initial_food)

        self.base_obs = np.zeros(
            (self.height, self.width, self.features), dtype=np.int8
        )

        self.base_obs[:, :, 0] = 1
        for r, c in self.walls:
            self.base_obs[r, c, 0] = 0
            self.base_obs[r, c, 4] = 1

        self._precompute_visibility_cache()
        self.reset()

    def reset(self):
        self.agents = self._initial_agents.copy()
        self.food_positions = self._initial_food.copy()
        self.steps = 0
        self.rewards = {0: 0, 1: 0}
        self.terminal = False

        return self._get_ego_centric_obs()

    def _place_agent(self, agent_id, position):
        self.agents[agent_id] = position

    def _get_freed_positions(self):
        occupied = self.food_positions.union(self.walls)
        freed = []
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) not in occupied:
                    freed.append((i, j))
        return freed

    def _get_agent_positions(self):
        return [self.agents[0], self.agents[1]]

    def _get_food_positions(self):
        return list(self.food_positions)

    def _get_wall_positions(self):
        return list(self.walls)

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

    def _precompute_visibility_cache(self):
        self._vis_cache = {}
        for r_agent in range(self.height):
            for c_agent in range(self.width):
                vis_map = np.zeros((self.height, self.width), dtype=np.int8)
                if (r_agent, c_agent) not in self.walls:
                    for r in range(self.height):
                        for c in range(self.width):
                            if (
                                max(abs(r - r_agent), abs(c - c_agent))
                                <= self.vision_radius
                            ):
                                if self._has_line_of_sight(r_agent, c_agent, r, c):
                                    vis_map[r, c] = 1
                self._vis_cache[(r_agent, c_agent)] = vis_map

    def get_visibility_map(self, agent_id):
        r_agent, c_agent = self.agents[agent_id]
        if r_agent is not None and c_agent is not None:
            return self._vis_cache[(r_agent, c_agent)].copy()
        return np.zeros((self.height, self.width), dtype=np.int8)

    def get_global_state(self):
        obs = np.zeros((self.height, self.width, self.features), dtype=np.int8)
        obs[:, :, 5] = 1  # All cells visible in global state
        for r, c in self.walls:
            obs[r, c, 4] = 1
        for r, c in self.food_positions:
            obs[r, c, 1] = 1
        if self.agents[0] is not None:
            r0, c0 = self.agents[0]
            obs[r0, c0, 2] = 1
        if self.agents[1] is not None:
            r1, c1 = self.agents[1]
            obs[r1, c1, 3] = 1
        for r in range(self.height):
            for c in range(self.width):
                if (
                    obs[r, c, 1] == 0
                    and obs[r, c, 2] == 0
                    and obs[r, c, 3] == 0
                    and obs[r, c, 4] == 0
                ):
                    obs[r, c, 0] = 1
        return obs

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
        self.agents[0] = A_pos
        self.agents[1] = B_pos
        return self._get_ego_centric_obs()

    def _get_action_space(self):
        return [0, 1, 2, 3]

    def _get_observations(self):
        observations = {}
        for agent_id in self.agents:
            obs = np.zeros((self.height, self.width, self.features), dtype=np.int8)
            vis_map = self.get_visibility_map(agent_id)
            obs[:, :, 5] = vis_map

            # Walls (known)
            for r, c in self.walls:
                obs[r, c, 4] = 1

            # Visible food
            for r, c in self.food_positions:
                if vis_map[r, c] == 1:
                    obs[r, c, 1] = 1

            # Visible Agent 0
            if self.agents[0] is not None:
                r0, c0 = self.agents[0]
                if vis_map[r0, c0] == 1:
                    obs[r0, c0, 2] = 1

            # Visible Agent 1
            if self.agents[1] is not None:
                r1, c1 = self.agents[1]
                if vis_map[r1, c1] == 1:
                    obs[r1, c1, 3] = 1

            # Empty visible tiles
            for r in range(self.height):
                for c in range(self.width):
                    if (
                        vis_map[r, c] == 1
                        and obs[r, c, 1] == 0
                        and obs[r, c, 2] == 0
                        and obs[r, c, 3] == 0
                        and obs[r, c, 4] == 0
                    ):
                        obs[r, c, 0] = 1

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

    def step(self, actions):
        rewards = {0: 0.0, 1: 0.0}
        new_positions = {}

        for agent_id, action in actions.items():
            r, c = self.agents[agent_id]

            if action == 0:  # Up
                r = max(0, r - 1)
            elif action == 1:  # Down
                r = min(self.height - 1, r + 1)
            elif action == 2:  # Left
                c = max(0, c - 1)
            elif action == 3:  # Right
                c = min(self.width - 1, c + 1)

            new_pos_tuple = (r, c)
            if new_pos_tuple in self.walls:
                new_positions[agent_id] = self.agents[agent_id]
            else:
                new_positions[agent_id] = new_pos_tuple

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

    @staticmethod
    def render_from_obs(obs):
        h, w = obs.shape[0], obs.shape[1]
        has_fog = obs.shape[2] >= 6
        render_grid = np.full((h, w), ".", dtype=str)
        for i in range(h):
            for j in range(w):
                if has_fog and obs[i, j, 5] == 0:
                    render_grid[i, j] = "?"  # Fog of War / Unobserved
                elif obs[i, j, 4] == 1:
                    render_grid[i, j] = "#"  # Wall
                elif obs[i, j, 1] == 1:
                    render_grid[i, j] = "F"  # Food
                elif obs[i, j, 2] == 1 and obs[i, j, 3] == 1:
                    render_grid[i, j] = "X"  # Both agents
                elif obs[i, j, 2] == 1:
                    render_grid[i, j] = "A"  # Agent 1 (Self)
                elif obs[i, j, 3] == 1:
                    render_grid[i, j] = "B"  # Agent 2 (Opponent)
        for row in render_grid:
            print(" ".join(row))
        print()

    def render(self, agent_id=0):
        obs = self._get_ego_centric_obs()[agent_id]
        self.render_from_obs(obs)

    def render_global(self):
        obs = self.get_global_state()
        self.render_from_obs(obs)


def a_star_path(start, goal, obstacles, h, w):
    # queue stores: (f_score, tie_breaker, (r, c), path)
    queue = []
    heapq.heappush(queue, (0, 0, start, []))

    g_costs = {start: 0}
    counter = 1  # Tie-breaker so heapq doesn't crash comparing tuples

    while queue:
        _, _, (r, c), path = heapq.heappop(queue)

        if (r, c) == goal:
            return path

        # 0: Up, 1: Down, 2: Left, 3: Right
        for dr, dc, action in [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]:
            nr, nc = r + dr, c + dc

            if 0 <= nr < h and 0 <= nc < w:
                if (nr, nc) not in obstacles:
                    new_cost = g_costs[(r, c)] + 1

                    # If we found a shorter path, or haven't visited this neighbor yet
                    if (nr, nc) not in g_costs or new_cost < g_costs[(nr, nc)]:
                        g_costs[(nr, nc)] = new_cost

                        # Manhattan distance heuristic
                        h_cost = abs(nr - goal[0]) + abs(nc - goal[1])
                        f_cost = new_cost + h_cost  # f = g + h

                        heapq.heappush(
                            queue, (f_cost, counter, (nr, nc), path + [action])
                        )
                        counter += 1

    return []  # No path found


def precompute_paths(obstacles: set, h: int, w: int):
    all_paths = {}
    inv_action = {0: 1, 1: 0, 2: 3, 3: 2}
    for r1 in range(h):
        for c1 in range(w):
            for r2 in range(h):
                for c2 in range(w):
                    start = (r1, c1)
                    goal = (r2, c2)
                    if (
                        start not in obstacles
                        and goal not in obstacles
                        and (start, goal) not in all_paths
                    ):
                        path = a_star_path(start, goal, obstacles, h, w)
                        all_paths[(start, goal)] = path
                        all_paths[(goal, start)] = [
                            inv_action[a]
                            # Reverse path and invert actions
                            for a in reversed(path)
                        ]
    print(
        f"Precomputed paths for all pairs of positions. Total pairs: {len(all_paths) // 2}"
    )
    return all_paths


class RandomAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def reset(self, initial_food=None, initial_opp_pos=None):
        pass

    def update_belief(self, observation):
        pass

    def select_action(self, observation, eval=False):
        return (
            np.random.randint(0, 4),
            None,
            np.zeros((observation.shape[0], observation.shape[1]), dtype=np.float32),
        )


def _parse_map_for_agent(agent_id, map_layout):
    if map_layout is None:
        raise ValueError(
            "map_layout must be provided for agent_id: {}".format(agent_id)
        )
    initial_food = set()
    initial_opp_pos = None
    for r, row in enumerate(map_layout):
        for c, char in enumerate(row):
            if char == "o":
                initial_food.add((r, c))
            elif char == "A" and agent_id == 1:
                initial_opp_pos = (r, c)
            elif char == "B" and agent_id == 0:
                initial_opp_pos = (r, c)
    return initial_food, initial_opp_pos


class SimpleAgent:
    def __init__(self, agent_id, precomputed_paths=None, map_layout=None):
        self.agent_id = agent_id
        self.cached_path = []
        self.current_target = None
        self.precomputed_paths = precomputed_paths
        self.initial_food, self.initial_opp_pos = _parse_map_for_agent(
            agent_id, map_layout
        )
        self.belief_food = None
        self.belief_opp_pos = None

    def reset(self, initial_food=None, initial_opp_pos=None):
        self.cached_path = []
        self.current_target = None
        self.belief_food = (
            set(initial_food) if initial_food is not None else set(self.initial_food)
        )
        self.belief_opp_pos = (
            initial_opp_pos if initial_opp_pos is not None else self.initial_opp_pos
        )

    def update_belief(self, observation):
        vis_map = (
            observation[:, :, 5]
            if observation.shape[2] >= 6
            else np.ones(observation.shape[:2], dtype=np.int8)
        )

        if self.belief_food is None:
            self.belief_food = set()
            food_pos_arr = np.argwhere(observation[:, :, 1] == 1)
            for p in food_pos_arr:
                self.belief_food.add(tuple(p))

        # Update food belief
        visible_indices = np.argwhere(vis_map == 1)
        for r, c in visible_indices:
            pos = (r, c)
            if observation[r, c, 1] == 1:
                self.belief_food.add(pos)
            elif pos in self.belief_food:
                self.belief_food.discard(pos)

        # Update opponent belief
        opp_pos_arr = np.argwhere(observation[:, :, 3] == 1)
        if len(opp_pos_arr) > 0:
            self.belief_opp_pos = tuple(opp_pos_arr[0])
        elif self.belief_opp_pos is not None:
            r_opp, c_opp = self.belief_opp_pos
            if vis_map[r_opp, c_opp] == 1:
                self.belief_opp_pos = None

    def get_subgoal_heatmap(self, observation):
        self.update_belief(observation)
        h, w = observation.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        food_positions = list(self.belief_food)

        if not food_positions:
            return heatmap

        if self.current_target in food_positions:
            # Target locked
            heatmap[self.current_target[0], self.current_target[1]] = 1.0
        else:
            # Uniform over choices in belief_food
            prob = 1.0 / len(food_positions)
            for f in food_positions:
                heatmap[f[0], f[1]] = prob

        return heatmap

    def select_action(self, observation, eval=False):
        my_channel = 2
        self.update_belief(observation)
        heatmap = self.get_subgoal_heatmap(observation)

        agent_pos_arr = np.argwhere(observation[:, :, my_channel] == 1)
        if len(agent_pos_arr) == 0:
            return np.random.randint(0, 4), None, heatmap
        my_pos = tuple(agent_pos_arr[0])

        food_positions = list(self.belief_food)
        if not food_positions:
            return np.random.randint(0, 4), None, heatmap

        if self.precomputed_paths is None:
            wall_pos_arr = np.argwhere(observation[:, :, 4] == 1)
            obstacles = set(tuple(p) for p in wall_pos_arr)
            self.precomputed_paths = precompute_paths(
                obstacles, observation.shape[0], observation.shape[1]
            )

        if self.current_target not in food_positions:
            random_index = np.random.randint(0, len(food_positions))
            self.current_target = food_positions[random_index]
            self.cached_path = []

        if not self.cached_path:
            if (my_pos, self.current_target) in self.precomputed_paths:
                self.cached_path = self.precomputed_paths[
                    (my_pos, self.current_target)
                ].copy()
            else:
                wall_pos_arr = np.argwhere(observation[:, :, 4] == 1)
                obstacles = set(tuple(p) for p in wall_pos_arr)
                self.cached_path = a_star_path(
                    my_pos,
                    self.current_target,
                    obstacles,
                    observation.shape[0],
                    observation.shape[1],
                )

        if self.cached_path:
            return self.cached_path.pop(0), None, heatmap
        else:
            return np.random.randint(0, 4), None, heatmap


class GreedySwitchAgent:
    """
    An advanced opponent. It goes for the absolute closest food in its belief map.
    If it knows/sees the other agent is closer to that food, it abandons it and switches to another.
    """

    def __init__(self, agent_id, precomputed_paths=None, map_layout=None):
        self.agent_id = agent_id
        self.cached_path = []
        self.current_target = None
        self.precomputed_paths = precomputed_paths
        self.initial_food, self.initial_opp_pos = _parse_map_for_agent(
            agent_id, map_layout
        )
        self.belief_food = None
        self.belief_opp_pos = None

    def reset(self, initial_food=None, initial_opp_pos=None):
        self.cached_path = []
        self.current_target = None
        self.belief_food = (
            set(initial_food) if initial_food is not None else set(self.initial_food)
        )
        self.belief_opp_pos = (
            initial_opp_pos if initial_opp_pos is not None else self.initial_opp_pos
        )

    def update_belief(self, observation):
        vis_map = (
            observation[:, :, 5]
            if observation.shape[2] >= 6
            else np.ones(observation.shape[:2], dtype=np.int8)
        )

        if self.belief_food is None:
            self.belief_food = set()
            food_pos_arr = np.argwhere(observation[:, :, 1] == 1)
            for p in food_pos_arr:
                self.belief_food.add(tuple(p))

        # Update food belief
        visible_indices = np.argwhere(vis_map == 1)
        for r, c in visible_indices:
            pos = (r, c)
            if observation[r, c, 1] == 1:
                self.belief_food.add(pos)
            elif pos in self.belief_food:
                self.belief_food.discard(pos)

        # Update opponent belief
        opp_pos_arr = np.argwhere(observation[:, :, 3] == 1)
        if len(opp_pos_arr) > 0:
            self.belief_opp_pos = tuple(opp_pos_arr[0])
        elif self.belief_opp_pos is not None:
            r_opp, c_opp = self.belief_opp_pos
            if vis_map[r_opp, c_opp] == 1:
                self.belief_opp_pos = None

    def get_subgoal_heatmap(self, observation):
        self.update_belief(observation)
        h, w = observation.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        agent_pos_arr = np.argwhere(observation[:, :, 2] == 1)
        food_positions = list(self.belief_food)

        if len(agent_pos_arr) == 0 or not food_positions:
            return heatmap

        my_pos = tuple(agent_pos_arr[0])
        opp_pos = self.belief_opp_pos

        if self.precomputed_paths is None:
            wall_pos_arr = np.argwhere(observation[:, :, 4] == 1)
            self.precomputed_paths = precompute_paths(
                set(tuple(p) for p in wall_pos_arr), h, w
            )

        dists = []
        for f in food_positions:
            my_dist = len(self.precomputed_paths.get((my_pos, f), []))
            if opp_pos is not None:
                opp_dist = len(self.precomputed_paths.get((opp_pos, f), []))
            else:
                opp_dist = float("inf")
            dists.append((my_dist, opp_dist, f))

        dists.sort(key=lambda x: x[0])
        min_my_dist = min(d[0] for d in dists)
        tie_foods = [d for d in dists if d[0] == min_my_dist]

        target_food = None
        for d in tie_foods:
            if self.current_target == d[2]:
                target_food = d[2]
                break

        if target_food is not None:
            chosen_dist = next(d for d in dists if d[2] == target_food)
            if chosen_dist[1] < chosen_dist[0]:
                safer_foods = [d for d in dists if d[0] <= d[1]]
                if safer_foods:
                    safer_foods.sort(key=lambda x: x[0])
                    target_food = safer_foods[0][2]
            heatmap[target_food[0], target_food[1]] = 1.0
        else:
            prob_per_tie = 1.0 / len(tie_foods)
            for d in tie_foods:
                potential_target = d[2]
                if d[1] < d[0]:
                    safer_foods = [sd for sd in dists if sd[0] <= sd[1]]
                    if safer_foods:
                        safer_foods.sort(key=lambda x: x[0])
                        potential_target = safer_foods[0][2]
                heatmap[potential_target[0], potential_target[1]] += prob_per_tie

        return heatmap

    def select_action(self, observation, eval=False):
        my_channel = 2
        self.update_belief(observation)
        heatmap = self.get_subgoal_heatmap(observation)
        agent_pos_arr = np.argwhere(observation[:, :, my_channel] == 1)

        if len(agent_pos_arr) == 0:
            return np.random.randint(0, 4), None, heatmap

        my_pos = tuple(agent_pos_arr[0])
        opp_pos = self.belief_opp_pos

        food_positions = list(self.belief_food)
        if not food_positions:
            return np.random.randint(0, 4), None, heatmap

        if self.precomputed_paths is None:
            wall_pos_arr = np.argwhere(observation[:, :, 4] == 1)
            obstacles = set(tuple(p) for p in wall_pos_arr)
            self.precomputed_paths = precompute_paths(
                obstacles, observation.shape[0], observation.shape[1]
            )

        dists = []
        for f in food_positions:
            my_dist = len(self.precomputed_paths.get((my_pos, f), []))
            if opp_pos is not None:
                opp_dist = len(self.precomputed_paths.get((opp_pos, f), []))
            else:
                opp_dist = float("inf")
            dists.append((my_dist, opp_dist, f))

        dists.sort(key=lambda x: x[0])
        min_my_dist = min(d[0] for d in dists)
        tie_foods = [d for d in dists if d[0] == min_my_dist]

        target_food = None
        for d in tie_foods:
            if self.current_target == d[2]:
                target_food = d[2]
                break

        if target_food is None:
            target_food = tie_foods[np.random.randint(len(tie_foods))][2]

        chosen_dist = next(d for d in dists if d[2] == target_food)
        if chosen_dist[1] < chosen_dist[0]:
            safer_foods = [d for d in dists if d[0] <= d[1]]
            if safer_foods:
                safer_foods.sort(key=lambda x: x[0])
                target_food = safer_foods[0][2]

        if self.current_target != target_food or not self.cached_path:
            self.current_target = target_food
            if (my_pos, target_food) in self.precomputed_paths:
                self.cached_path = self.precomputed_paths[(my_pos, target_food)].copy()
            else:
                wall_pos_arr = np.argwhere(observation[:, :, 4] == 1)
                obstacles = set(tuple(p) for p in wall_pos_arr)
                self.cached_path = a_star_path(
                    my_pos,
                    target_food,
                    obstacles,
                    observation.shape[0],
                    observation.shape[1],
                )

        if self.cached_path:
            return self.cached_path.pop(0), None, heatmap
        else:
            return np.random.randint(0, 4), None, heatmap


class StalkerAgent:
    """
    A Hyper-Reactive Interceptor. It identifies the nearest food in its belief map
    where it has a positional advantage over the opponent, races there, and loiters 1 tile away.
    """

    def __init__(self, agent_id, precomputed_paths=None, map_layout=None):
        self.agent_id = agent_id
        self.precomputed_paths = precomputed_paths
        self.initial_food, self.initial_opp_pos = _parse_map_for_agent(
            agent_id, map_layout
        )
        self.belief_food = None
        self.belief_opp_pos = None

    def reset(self, initial_food=None, initial_opp_pos=None):
        self.belief_food = (
            set(initial_food) if initial_food is not None else set(self.initial_food)
        )
        self.belief_opp_pos = (
            initial_opp_pos if initial_opp_pos is not None else self.initial_opp_pos
        )

    def update_belief(self, observation):
        vis_map = (
            observation[:, :, 5]
            if observation.shape[2] >= 6
            else np.ones(observation.shape[:2], dtype=np.int8)
        )

        if self.belief_food is None:
            self.belief_food = set()
            food_pos_arr = np.argwhere(observation[:, :, 1] == 1)
            for p in food_pos_arr:
                self.belief_food.add(tuple(p))

        # Update food belief
        visible_indices = np.argwhere(vis_map == 1)
        for r, c in visible_indices:
            pos = (r, c)
            if observation[r, c, 1] == 1:
                self.belief_food.add(pos)
            elif pos in self.belief_food:
                self.belief_food.discard(pos)

        # Update opponent belief
        opp_pos_arr = np.argwhere(observation[:, :, 3] == 1)
        if len(opp_pos_arr) > 0:
            self.belief_opp_pos = tuple(opp_pos_arr[0])
        elif self.belief_opp_pos is not None:
            r_opp, c_opp = self.belief_opp_pos
            if vis_map[r_opp, c_opp] == 1:
                self.belief_opp_pos = None

    def get_subgoal_heatmap(self, observation):
        self.update_belief(observation)
        h, w = observation.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        my_pos_arr = np.argwhere(observation[:, :, 2] == 1)
        food_positions = list(self.belief_food)

        if len(my_pos_arr) == 0 or not food_positions:
            return heatmap

        my_pos = tuple(my_pos_arr[0])
        opp_pos = self.belief_opp_pos

        if self.precomputed_paths is None:
            wall_pos_arr = np.argwhere(observation[:, :, 4] == 1)
            self.precomputed_paths = precompute_paths(
                set(tuple(p) for p in wall_pos_arr), h, w
            )

        if opp_pos is None:
            unseen_pos_arr = np.argwhere(observation[:, :, 5] == 0)
            if len(unseen_pos_arr) > 0:
                unseen_dists = []
                for p in unseen_pos_arr:
                    pt = tuple(p)
                    s_path = self.precomputed_paths.get((my_pos, pt), [])
                    if len(s_path) > 0:
                        unseen_dists.append((len(s_path), pt))
                if unseen_dists:
                    min_s_dist = min(d[0] for d in unseen_dists)
                    tie_unseen = [pt for sd, pt in unseen_dists if sd == min_s_dist]
                    prob = 1.0 / len(tie_unseen)
                    for pt in tie_unseen:
                        heatmap[pt[0], pt[1]] = prob
                    return heatmap

            greedy_foods = []
            for f in food_positions:
                s_dist = len(self.precomputed_paths.get((my_pos, f), [])) or float(
                    "inf"
                )
                if s_dist != float("inf"):
                    greedy_foods.append((s_dist, f))
            if greedy_foods:
                min_s_dist = min(d[0] for d in greedy_foods)
                tie_foods = [f for sd, f in greedy_foods if sd == min_s_dist]
                prob = 1.0 / len(tie_foods)
                for f in tie_foods:
                    heatmap[f[0], f[1]] += prob
            return heatmap

        winnable_foods = []
        for f in food_positions:
            e_dist = len(self.precomputed_paths.get((opp_pos, f), [])) or float("inf")
            s_dist = len(self.precomputed_paths.get((my_pos, f), [])) or float("inf")
            if s_dist <= e_dist and s_dist != float("inf"):
                winnable_foods.append((e_dist, s_dist, f))

        if winnable_foods:
            winnable_foods.sort(key=lambda x: x[0])
            min_e_dist = winnable_foods[0][0]
            tie_foods = [f for ed, sd, f in winnable_foods if ed == min_e_dist]

            prob = 1.0 / len(tie_foods)
            for f in tie_foods:
                heatmap[f[0], f[1]] += prob
        else:
            greedy_foods = []
            for f in food_positions:
                s_dist = len(self.precomputed_paths.get((my_pos, f), [])) or float(
                    "inf"
                )
                if s_dist != float("inf"):
                    greedy_foods.append((s_dist, f))

            if greedy_foods:
                greedy_foods.sort(key=lambda x: x[0])
                min_s_dist = greedy_foods[0][0]
                tie_foods = [f for sd, f in greedy_foods if sd == min_s_dist]

                prob = 1.0 / len(tie_foods)
                for f in tie_foods:
                    heatmap[f[0], f[1]] += prob

        return heatmap

    def select_action(self, observation, eval=False):
        my_channel = 2
        self.update_belief(observation)
        heatmap = self.get_subgoal_heatmap(observation)

        my_pos_arr = np.argwhere(observation[:, :, my_channel] == 1)
        if len(my_pos_arr) == 0:
            return np.random.randint(0, 4), None, heatmap

        my_pos = tuple(my_pos_arr[0])
        opp_pos = self.belief_opp_pos

        food_positions = list(self.belief_food)
        if not food_positions:
            return np.random.randint(0, 4), None, heatmap

        if self.precomputed_paths is None:
            wall_pos_arr = np.argwhere(observation[:, :, 4] == 1)
            obstacles = set(tuple(p) for p in wall_pos_arr)
            self.precomputed_paths = precompute_paths(
                obstacles, observation.shape[0], observation.shape[1]
            )

        if opp_pos is None:
            unseen_pos_arr = np.argwhere(observation[:, :, 5] == 0)
            if len(unseen_pos_arr) > 0:
                unseen_dists = []
                for p in unseen_pos_arr:
                    pt = tuple(p)
                    s_path = self.precomputed_paths.get((my_pos, pt), [])
                    if len(s_path) > 0:
                        unseen_dists.append((len(s_path), pt, s_path))
                if unseen_dists:
                    unseen_dists.sort(key=lambda x: x[0])
                    min_s_dist = unseen_dists[0][0]
                    tie_unseen = [d for d in unseen_dists if d[0] == min_s_dist]
                    chosen = tie_unseen[np.random.randint(len(tie_unseen))]
                    return chosen[2][0], None, heatmap

            greedy_foods = []
            for f in food_positions:
                s_path = self.precomputed_paths.get((my_pos, f), [])
                s_dist = len(s_path) if len(s_path) > 0 else float("inf")
                if s_dist != float("inf"):
                    greedy_foods.append((s_dist, f, s_path))
            if greedy_foods:
                greedy_foods.sort(key=lambda x: x[0])
                min_s_dist = greedy_foods[0][0]
                tie_foods = [d for d in greedy_foods if d[0] == min_s_dist]
                chosen = tie_foods[np.random.randint(len(tie_foods))]
                return chosen[2][0], None, heatmap
            return np.random.randint(0, 4), None, heatmap

        winnable_foods = []
        for f in food_positions:
            e_path = self.precomputed_paths.get((opp_pos, f), [])
            e_dist = len(e_path) if len(e_path) > 0 else float("inf")
            s_path = self.precomputed_paths.get((my_pos, f), [])
            s_dist = len(s_path) if len(s_path) > 0 else float("inf")

            if s_dist <= e_dist and s_dist != float("inf"):
                winnable_foods.append((e_dist, s_dist, f))

        if winnable_foods:
            winnable_foods.sort(key=lambda x: x[0])
            min_e_dist = winnable_foods[0][0]

            tie_foods = [f for ed, sd, f in winnable_foods if ed == min_e_dist]
            target_food = tie_foods[np.random.randint(len(tie_foods))]

            s_path = self.precomputed_paths.get((my_pos, target_food), [])
            s_dist = len(s_path)

            if s_dist == 1 and min_e_dist > 2:
                wall_pos_arr = np.argwhere(observation[:, :, 4] == 1)
                obstacles = set(tuple(p) for p in wall_pos_arr)

                for action, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                    nr, nc = my_pos[0] + dr, my_pos[1] + dc
                    if (nr, nc) in obstacles:
                        return action, None, heatmap
                return np.random.randint(0, 4), None, heatmap
        else:
            greedy_foods = []
            for f in food_positions:
                s_path = self.precomputed_paths.get((my_pos, f), [])
                s_dist = len(s_path) if len(s_path) > 0 else float("inf")
                if s_dist != float("inf"):
                    greedy_foods.append((s_dist, f))

            if greedy_foods:
                greedy_foods.sort(key=lambda x: x[0])
                min_s_dist = greedy_foods[0][0]
                tie_foods = [f for sd, f in greedy_foods if sd == min_s_dist]
                target_food = tie_foods[np.random.randint(len(tie_foods))]
            else:
                return np.random.randint(0, 4), None, heatmap

        p_path = self.precomputed_paths.get((my_pos, target_food), [])
        if p_path:
            return p_path[0], None, heatmap
        else:
            return np.random.randint(0, 4), None, heatmap


class ChameleonAgent:
    """
    Opponent that switches between Simple and Greedy.
    """

    def __init__(self, agent_id, precomputed_paths=None, map_layout=None):
        self.agent_id = agent_id
        self.simple_agent = SimpleAgent(agent_id, precomputed_paths, map_layout)
        self.greedy_agent = GreedySwitchAgent(agent_id, precomputed_paths, map_layout)
        self.current_persona = "greedy"

    def reset(self, initial_food=None, initial_opp_pos=None):
        self.simple_agent.reset(initial_food, initial_opp_pos)
        self.greedy_agent.reset(initial_food, initial_opp_pos)

    def update_belief(self, observation):
        self.simple_agent.update_belief(observation)
        self.greedy_agent.update_belief(observation)

    def get_subgoal_heatmap(self, observation):
        simple_hm = self.simple_agent.get_subgoal_heatmap(observation)
        greedy_hm = self.greedy_agent.get_subgoal_heatmap(observation)
        return (0.3 * simple_hm) + (0.7 * greedy_hm)

    def select_action(self, observation, eval=False):
        heatmap = self.get_subgoal_heatmap(observation)
        new_persona = "simple" if np.random.rand() < 0.3 else "greedy"

        if new_persona != self.current_persona:
            b_food = (
                self.greedy_agent.belief_food
                if self.current_persona == "greedy"
                else self.simple_agent.belief_food
            )
            b_opp = (
                self.greedy_agent.belief_opp_pos
                if self.current_persona == "greedy"
                else self.simple_agent.belief_opp_pos
            )
            self.simple_agent.reset(b_food, b_opp)
            self.greedy_agent.reset(b_food, b_opp)
            self.current_persona = new_persona

        if self.current_persona == "simple":
            action, _, _ = self.simple_agent.select_action(observation, eval)
        else:
            action, _, _ = self.greedy_agent.select_action(observation, eval)

        return action, None, heatmap

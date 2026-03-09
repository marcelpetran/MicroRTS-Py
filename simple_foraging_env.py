from turtle import back

import numpy as np
from collections import deque
from maps import *


class SimpleForagingEnv:
  def __init__(self, max_steps=50, map_layout=MAP_1):
    self.map_layout = map_layout
    self.grid_size = len(map_layout)
    self.num_agents = 2
    # 0: empty, 1: food, 2: agent1, 3: agent2, 4: wall
    self.features = 5
    self.max_steps = max_steps
    self.action_space = self._get_action_space()

    self._initial_agents = {0: None, 1: None}
    self._initial_food = set()
    self.walls = set()

    for i, row in enumerate(self.map_layout):
      for j, char in enumerate(row):
        pos = (i, j)
        if char == '#':
          self.walls.add(pos)
        elif char == 'o':
          self._initial_food.add(pos)
        elif char == 'A':
          self._initial_agents[0] = pos
        elif char == 'B':
          self._initial_agents[1] = pos

    self.num_food = len(self._initial_food)

    self.base_obs = np.zeros(
      (self.grid_size, self.grid_size, self.features), dtype=int)

    self.base_obs[:, :, 0] = 1
    for r, c in self.walls:
      self.base_obs[r, c, 0] = 0
      self.base_obs[r, c, 4] = 1
    self.reset()

  def reset(self):
    self.agents = self._initial_agents.copy()
    self.food_positions = self._initial_food.copy()
    self.steps = 0
    self.rewards = {0: 0, 1: 0}
    self.terminal = False

    return self._get_observations()

  def _place_agent(self, agent_id, position):
    self.agents[agent_id] = position

  def _get_freed_positions(self):
    occupied = set(self.agents.values()).union(
      self.food_positions).union(self.walls)
    freed = []
    for i in range(self.grid_size):
      for j in range(self.grid_size):
        if (i, j) not in occupied:
          freed.append((i, j))
    return freed

  def _get_agent_positions(self):
    return [self.agents[0], self.agents[1]]

  def _get_food_positions(self):
    return list(self.food_positions)

  def _get_wall_positions(self):
    return list(self.walls)

  def reset_random_spawn(self, agent_id):
    _ = self.reset()

    # Remove a random food
    if np.random.rand() > 0.5:
      food_list = list(self.food_positions)
      if len(food_list) > 0:
        removed_food = food_list[np.random.randint(len(food_list))]
        self.food_positions.remove(removed_food)

    freed = self._get_freed_positions()
    pos = freed[np.random.randint(0, len(freed))]
    self.agents[agent_id] = pos
    return self._get_observations()

  def _get_action_space(self):
    return [0, 1, 2, 3]

  def _get_observations(self):
    observations = {}
    for agent_id in self.agents:
      obs = self.base_obs.copy()

      for r, c in self.food_positions:
        obs[r, c, 0] = 0
        obs[r, c, 1] = 1

      r0, c0 = self.agents[0]
      obs[r0, c0, 0] = 0
      obs[r0, c0, 2] = 1

      r1, c1 = self.agents[1]
      obs[r1, c1, 0] = 0
      obs[r1, c1, 3] = 1

      observations[agent_id] = obs
    return observations

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
        r = min(self.grid_size - 1, r + 1)
      elif action == 2:  # Left
        c = max(0, c - 1)
      elif action == 3:  # Right
        c = min(self.grid_size - 1, c + 1)

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

    return self._get_observations(), rewards, self._check_terminal(), {}

  @staticmethod
  def render_from_obs(obs):
    grid_size = obs.shape[0]
    render_grid = np.full((grid_size, grid_size), '.', dtype=str)
    for i in range(grid_size):
      for j in range(grid_size):
        if obs[i, j, 4] == 1:
          render_grid[i, j] = '#'  # Wall
        elif obs[i, j, 1] == 1:
          render_grid[i, j] = 'F'  # Food
        elif obs[i, j, 2] == 1 and obs[i, j, 3] == 1:
          render_grid[i, j] = 'X'  # Both agents
        elif obs[i, j, 2] == 1:
          render_grid[i, j] = 'A'  # Agent 1
        elif obs[i, j, 3] == 1:
          render_grid[i, j] = 'B'  # Agent 2
    for row in render_grid:
      print(' '.join(row))
    print()


def bfs_path(start, goal, obstacles, grid_size):
  queue = deque([(start, [])])
  visited = set([start])

  while queue:
    (r, c), path = queue.popleft()
    if (r, c) == goal:
      return path

    # 0: Up, 1: Down, 2: Left, 3: Right
    for dr, dc, action in [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]:
      nr, nc = r + dr, c + dc
      if 0 <= nr < grid_size and 0 <= nc < grid_size:
        if (nr, nc) not in obstacles and (nr, nc) not in visited:
          visited.add((nr, nc))
          queue.append(((nr, nc), path + [action]))
  return []  # No path found


class RandomAgent:
  def __init__(self, agent_id):
    self.agent_id = agent_id

  def reset(self): pass

  def select_action(self, observation, eval=False):
    return np.random.randint(0, 4)


class SimpleAgent:
  def __init__(self, agent_id):
    self.agent_id = agent_id
    self.cached_path = []
    self.current_target = None

  def reset(self):
    self.cached_path = []
    self.current_target = None

  def select_action(self, observation, eval=False):
    agent_pos_arr = np.argwhere(observation[:, :, 2 + self.agent_id] == 1)
    if len(agent_pos_arr) == 0:
      return np.random.randint(0, 4)
    my_pos = tuple(agent_pos_arr[0])

    food_positions = [tuple(p) for p in np.argwhere(observation[:, :, 1] == 1)]
    if not food_positions:
      return np.random.randint(0, 4)

    if self.current_target not in food_positions:
      random_index = np.random.randint(0, len(food_positions))
      self.current_target = food_positions[random_index]
      self.cached_path = []

    if not self.cached_path:
      wall_pos_arr = np.argwhere(observation[:, :, 4] == 1)
      obstacles = set(tuple(p) for p in wall_pos_arr)

      self.cached_path = bfs_path(
        my_pos, self.current_target, obstacles, observation.shape[0])

    if self.cached_path:
      return self.cached_path.pop(0)
    else:
      return np.random.randint(0, 4)


class GreedySwitchAgent:
  """
  An advanced opponent. It goes for the absolute closest food. 
  If it realizes the other agent is closer to that food, it abandons it and switches to another.
  """

  def __init__(self, agent_id):
    self.agent_id = agent_id
    self.cached_path = []
    self.current_target = None

  def reset(self):
    self.cached_path = []
    self.current_target = None

  def select_action(self, observation, eval=False):
    agent_pos_arr = np.argwhere(observation[:, :, 2 + self.agent_id] == 1)
    opp_pos_arr = np.argwhere(observation[:, :, 2 + (1 - self.agent_id)] == 1)

    if len(agent_pos_arr) == 0 or len(opp_pos_arr) == 0:
      return np.random.randint(0, 4)

    my_pos = tuple(agent_pos_arr[0])
    opp_pos = tuple(opp_pos_arr[0])

    food_positions = [tuple(p) for p in np.argwhere(observation[:, :, 1] == 1)]
    if not food_positions:
      return np.random.randint(0, 4)

    # Compute distances to all food and sort by my distance
    dists = []
    for f in food_positions:
      my_dist = abs(my_pos[0] - f[0]) + abs(my_pos[1] - f[1])
      opp_dist = abs(opp_pos[0] - f[0]) + abs(opp_pos[1] - f[1])
      dists.append((my_dist, opp_dist, f))

    dists.sort(key=lambda x: x[0])
    idx = 1
    # pick random food if tie in distances
    for i in range(1, len(dists)):
      if dists[i][0] != dists[0][0]:
        idx = i
        break
    if idx > 1:
      tie_foods = dists[:idx]
      target_food = tie_foods[np.random.randint(len(tie_foods))][2]
    else:
      target_food = dists[0][2]

    if len(dists) > 1:
      primary = dists[0]
      backup = dists[1]
      if primary[1] < primary[0]:
        # Opponent is closer to the primary target food
        # Check tie foods to see if there's a better alternative
        for d in tie_foods:
          if d[1] >= d[0]:
            # Found a tie food that is closer
            backup = d
            break
        # If no tie food is better, backup will just be the second closest food
        target_food = backup[2]

    # recompute path to the chosen target food only when needed
    if self.current_target != target_food or not self.cached_path:
      self.current_target = target_food

      wall_pos_arr = np.argwhere(observation[:, :, 4] == 1)
      obstacles = set(tuple(p) for p in wall_pos_arr)

      self.cached_path = bfs_path(
        my_pos, target_food, obstacles, observation.shape[0])

    if self.cached_path:
      return self.cached_path.pop(0)
    else:
      return np.random.randint(0, 4)

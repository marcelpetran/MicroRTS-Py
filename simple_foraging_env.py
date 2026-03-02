import numpy as np
from collections import deque


class SimpleForagingEnv:
  def __init__(self, grid_size=11, max_steps=50):
    self.grid_size = grid_size
    self.num_agents = 2
    self.num_food = 3
    # 0: empty, 1: food, 2: agent1, 3: agent2, 4: wall
    self.features = 5
    self.max_steps = max_steps
    self.rewards = {0: 0, 1: 0}
    self.terminal = False
    self.action_space = self._get_action_space()
    self.reset()

  def reset(self):
    self.agents = {}
    self.num_food = 3
    self.food_positions = set()
    self.walls = set()
    self.steps = 0

    mid = self.grid_size // 2

    # 1. Zoned Agent Placement (Left and Right edges)
    self.agents[0] = (np.random.randint(0, self.grid_size), 0)
    self.agents[1] = (np.random.randint(0, self.grid_size), self.grid_size - 1)

    # 2. Strategic Walls (Horizontal barrier with a center gap)
    # Creates walls at row 5, cols 2,3,4 and 6,7,8. Leaves col 5 open.
    for c in range(2, self.grid_size - 2):
      if c != mid:  # Leave a gap in the middle
        self.walls.add((c, mid))

    # 3. Strategic Food Placement
    # Food 1: Left Safe Food. Ensure it doesn't spawn IN the wall (r=mid)
    r1 = np.random.choice([r for r in range(self.grid_size) if r != mid])
    self.food_positions.add((r1, 2))

    # Food 2: Right Safe Food. Ensure it doesn't spawn IN the wall.
    r2 = np.random.choice([r for r in range(self.grid_size) if r != mid])
    self.food_positions.add((r2, self.grid_size - 3))

    # Food 3: Central Food spawns exactly in the wall gap
    self.food_positions.add((mid, mid))

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
    return list(self.agents.values())

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
    for agent_id, pos in self.agents.items():
      obs_grid = np.zeros(
        (self.grid_size, self.grid_size, self.features), dtype=int)
      for i in range(self.grid_size):
        for j in range(self.grid_size):
          if (i, j) in self.walls:
            obs_grid[i, j, 4] = 1  # wall
          elif (i, j) in self.food_positions:
            obs_grid[i, j, 1] = 1  # food
          elif (i, j) == self.agents[0]:
            obs_grid[i, j, 2] = 1  # agent1
          elif (i, j) == self.agents[1]:
            obs_grid[i, j, 3] = 1  # agent2
          else:
            obs_grid[i, j, 0] = 1  # empty
      observations[agent_id] = obs_grid
    return observations

  def _check_terminal(self):
    if self.steps >= self.max_steps or len(self.food_positions) == 0:
      self.terminal = True
    return self.terminal

  def step(self, actions):
    rewards = {agent_id: 0 for agent_id in range(self.num_agents)}
    new_positions = {}

    for agent_id, action in actions.items():
      current_pos = self.agents[agent_id]
      new_pos = list(current_pos)

      if action == 0:  # Up
        new_pos[0] = max(0, new_pos[0] - 1)
      elif action == 1:  # Down
        new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)
      elif action == 2:  # Left
        new_pos[1] = max(0, new_pos[1] - 1)
      elif action == 3:  # Right
        new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)

      new_pos_tuple = tuple(new_pos)
      # Check wall collision
      if new_pos_tuple in self.walls:
        new_positions[agent_id] = current_pos  # Stay still if hit wall
      else:
        new_positions[agent_id] = new_pos_tuple

    self.agents = new_positions
    self.steps += 1

    # Check for food collection
    for agent_id, pos in self.agents.items():
      if pos in self.food_positions:
        rewards[agent_id] += 1
        self.food_positions.remove(pos)

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
    self.target_idx = np.random.randint(0, 3)

  def reset(self):
    self.target_idx = np.random.randint(0, 3)

  def select_action(self, observation, eval=False):
    grid_size = observation.shape[0]
    agent_pos = None
    food_positions = []
    obstacles = set()

    for i in range(grid_size):
      for j in range(grid_size):
        if observation[i, j, 2 + self.agent_id] == 1:
          agent_pos = (i, j)
        if observation[i, j, 2 + (1 - self.agent_id)] == 1:
          obstacles.add((i, j))  # Avoid other agent
        if observation[i, j, 4] == 1:
          obstacles.add((i, j))  # Avoid walls
        if observation[i, j, 1] == 1:
          food_positions.append((i, j))

    if not food_positions or not agent_pos:
      return np.random.randint(0, 4)

    food_positions.sort(key=lambda x: x[1])
    
    # Pick the target food based on the agent's assigned role
    idx = min(self.target_idx, len(food_positions) - 1)
    target_food = food_positions[idx]

    action_seq = bfs_path(agent_pos, target_food, obstacles, grid_size)
    return action_seq[0] if action_seq else np.random.randint(0, 4)


class GreedySwitchAgent:
  """
  An advanced opponent. It goes for the absolute closest food. 
  If it realizes the other agent is closer to that food, it abandons it and switches to another.
  """

  def __init__(self, agent_id):
    self.agent_id = agent_id

  def reset(self): pass

  def select_action(self, observation, eval=False):
    grid_size = observation.shape[0]
    my_pos, opp_pos = None, None
    food_positions = []
    obstacles = set()

    for i in range(grid_size):
      for j in range(grid_size):
        if observation[i, j, 2 + self.agent_id] == 1:
          my_pos = (i, j)
        if observation[i, j, 2 + (1 - self.agent_id)] == 1:
          opp_pos = (i, j)
          obstacles.add((i, j))
        if observation[i, j, 4] == 1:
          obstacles.add((i, j))
        if observation[i, j, 1] == 1:
          food_positions.append((i, j))

    if not food_positions or not my_pos or not opp_pos:
      return np.random.randint(0, 4)

    # Calculate Manhattan distances
    dists = []
    for f in food_positions:
      my_dist = abs(my_pos[0] - f[0]) + abs(my_pos[1] - f[1])
      opp_dist = abs(opp_pos[0] - f[0]) + abs(opp_pos[1] - f[1])
      dists.append((my_dist, opp_dist, f))

    # Sort by my distance
    dists.sort(key=lambda x: x[0])

    target_food = dists[0][2]  # Default: closest food

    # Switch logic: If opponent is strictly closer to my target, and there's another food, switch!
    if len(dists) > 1:
      primary = dists[0]
      backup = dists[1]
      # If opponent is closer to primary, but I am closer to backup (or at least it's safer)
      if primary[1] < primary[0]:
        target_food = backup[2]

    action_seq = bfs_path(my_pos, target_food, obstacles, grid_size)
    return action_seq[0] if action_seq else np.random.randint(0, 4)

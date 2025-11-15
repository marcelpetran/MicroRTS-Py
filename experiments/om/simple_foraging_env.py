import numpy as np


class SimpleForagingEnv:
  def __init__(self, grid_size=5, max_steps=50):
    self.grid_size = grid_size
    self.num_agents = 2
    self.num_food = 2
    self.features = 4  # empty, food, agent1, agent2
    self.max_steps = max_steps
    self.rewards = {0: 0, 1: 0}
    self.terminal = False
    self.action_space = self._get_action_space()
    self.reset()

  def reset(self):
    self.agents = {}
    self.food_positions = set()
    self.steps = 0

    # Place agents
    self.agents[0] = (self.grid_size // 2, 0)
    self.agents[1] = (self.grid_size // 2, self.grid_size - 1)

    # Place food
    pos1 = (0, self.grid_size // 2)
    pos2 = (self.grid_size - 1, self.grid_size // 2)
    self.food_positions.add(pos1)
    self.food_positions.add(pos2)

    self.rewards = {0: 0, 1: 0}
    self.terminal = False

    # obs, info
    return self._get_observations()
  
  def _place_agent(self, agent_id, position):
    self.agents[agent_id] = position
  
  def _get_freed_positions(self):
    occupied = set(self.agents.values()).union(self.food_positions)
    freed = []
    for i in range(self.grid_size):
      for j in range(self.grid_size):
        if (i, j) not in occupied:
          freed.append((i, j))
    return freed
  
  def reset_random_spawn(self, agent_id):
    _ = self.reset()
    freed = self._get_freed_positions()
    pos = freed[np.random.randint(0, len(freed))]
    self.agents[agent_id] = pos
    return self._get_observations()

  def _get_action_space(self):
    # 4 actions: up, down, left, right
    return [i for i in range(4)]

  def _get_observations(self):
    observations = {}
    for agent_id, pos in self.agents.items():
      obs_grid = np.zeros(
        (self.grid_size, self.grid_size, self.features), dtype=int)
      for i in range(self.grid_size):
        for j in range(self.grid_size):
          if (i, j) in self.food_positions:
            obs_grid[i, j, 1] = 1  # food
          if (i, j) == self.agents[0]:
            obs_grid[i, j, 2] = 1  # agent1
          if (i, j) == self.agents[1]:
            obs_grid[i, j, 3] = 1  # agent2
          if (i, j) not in self.food_positions and (i, j) != self.agents[0] and (i, j) != self.agents[1]:
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

    # Move agents
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
        new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)  # Right
      # Check for collisions with other agents
      if tuple(new_pos) not in new_positions.values():
        new_positions[agent_id] = tuple(new_pos)
      else:
        new_positions[agent_id] = current_pos  # Stay in place if collision

    self.agents = new_positions
    self.steps += 1
    # Check for food collection
    for agent_id, pos in self.agents.items():
      if pos in self.food_positions:
        rewards[agent_id] += 10
        self.food_positions.remove(pos)
      else:
        rewards[agent_id] -= 0.1  # Penalty for step
    # obs, rewards, done, info
    return self._get_observations(), rewards, self._check_terminal(), {}

  @staticmethod
  def render_from_obs(obs):
    """
    Render the grid from the observation
    'F' = food
    'A' = agent 1
    'B' = agent 2
    '.' = empty
    """
    grid_size = obs.shape[0]
    render_grid = np.full((grid_size, grid_size), '.', dtype=str)
    for i in range(grid_size):
      for j in range(grid_size):
        if obs[i, j, 1] == 1:
          render_grid[i, j] = 'F'  # Food
        if obs[i, j, 2] == 1:
          render_grid[i, j] = 'A'  # Agent 1
        if obs[i, j, 3] == 1:
          render_grid[i, j] = 'B'  # Agent 2
    for row in render_grid:
      print(' '.join(row))
    print()


class RandomAgent:
  def __init__(self, agent_id):
    self.agent_id = agent_id

  def reset(self):
    pass

  def select_action(self, observation):
    return np.random.randint(0, 4)  # 4 actions: up, down, left, right


class SimpleAgent:
  """
  Simple agent, that decides by coin if he goes for the top or bottom food
  """

  def __init__(self, agent_id, always_go_top=False):
    self.agent_id = agent_id
    # agent randomly decides to go for top or bottom food
    self.going_for_top = True if np.random.rand() > 0.5 else False
    self.always_go_top = always_go_top
    if self.always_go_top:
      self.going_for_top = True

  def reset(self):
    self.going_for_top = True if np.random.rand() > 0.5 else False
    if self.always_go_top:
      self.going_for_top = True

  def find_food(self, food_positions):
    if not food_positions:
      return None
    if self.going_for_top == True:
      # food with smallest row index
      return min(food_positions, key=lambda x: x[0])
    else:
      # food with largest row index
      return max(food_positions, key=lambda x: x[0])

  def find_path(self, start, goal, obstacles):
    path = []
    action_seq = []
    current = start
    action = 0  # 0: up, 1: down, 2: left, 3: right
    # also consider obstacles (other agent)
    while current != goal:
      if current[0] < goal[0] and (current[0] + 1, current[1]) not in obstacles:
        # move down
        action = 1
        current = (current[0] + 1, current[1])
      elif current[0] > goal[0] and (current[0] - 1, current[1]) not in obstacles:
        # move up
        action = 0
        current = (current[0] - 1, current[1])
      elif current[1] < goal[1] and (current[0], current[1] + 1) not in obstacles:
        # move right
        action = 3
        current = (current[0], current[1] + 1)
      elif current[1] > goal[1] and (current[0], current[1] - 1) not in obstacles:
        # move left
        action = 2
        current = (current[0], current[1] - 1)
      else:
        break  # No valid move, break to avoid infinite loop
      path.append(current)
      action_seq.append(action)
    return path, action_seq

  def select_action(self, observation):
    agent_pos = None
    food_positions = []
    obstacle = []
    for i in range(observation.shape[0]):
      for j in range(observation.shape[1]):
        if observation[i, j, 2 + self.agent_id] == 1:
          agent_pos = (i, j)
        if observation[i, j, 2 + (1 - self.agent_id)] == 1:
          obstacle.append((i, j))
        if observation[i, j, 1] == 1:
          food_positions.append((i, j))
    if not food_positions:
      return np.random.randint(0, 4)  # No food left, random action
    target_food = self.find_food(food_positions)
    if target_food is None:
      return np.random.randint(0, 4)  # No food found, random action

    path, action_seq = self.find_path(agent_pos, target_food, obstacle)
    if action_seq:
      return action_seq[0]  # Take the first step in the path
    else:
      return np.random.randint(0, 4)  # random action, should not happen

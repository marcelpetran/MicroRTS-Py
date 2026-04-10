import heapq

import numpy as np
from seaborn import heatmap
from maps import *


class SimpleForagingEnv:
  def __init__(self, max_steps=50, map_layout=MAP_1):
    self.map_layout = map_layout
    self.height = len(map_layout)
    self.width = len(map_layout[0])
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
      (self.height, self.width, self.features), dtype=np.int8)

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
    render_grid = np.full((h, w), '.', dtype=str)
    for i in range(h):
      for j in range(w):
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

  def render(self):
    obs = self._get_observations()[0]
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

            heapq.heappush(queue, (f_cost, counter, (nr, nc), path + [action]))
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
          if start not in obstacles and goal not in obstacles and (start, goal) not in all_paths:
            path = a_star_path(start, goal, obstacles, h, w)
            all_paths[(start, goal)] = path
            all_paths[(goal, start)] = [inv_action[a]
                                        # Reverse path and invert actions
                                        for a in reversed(path)]
  print(
    f"Precomputed paths for all pairs of positions. Total pairs: {len(all_paths) // 2}")
  return all_paths


class RandomAgent:
  def __init__(self, agent_id):
    self.agent_id = agent_id

  def reset(self): pass

  def select_action(self, observation, eval=False):
    return np.random.randint(0, 4), None, np.zeros((observation.shape[0], observation.shape[1]), dtype=np.float32)


class SimpleAgent:
  def __init__(self, agent_id, precomputed_paths=None):
    self.agent_id = agent_id
    self.cached_path = []
    self.current_target = None
    self.precomputed_paths = precomputed_paths

  def reset(self):
    self.cached_path = []
    self.current_target = None

  def get_subgoal_heatmap(self, observation):
    h, w = observation.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    food_positions = [tuple(p) for p in np.argwhere(observation[:, :, 1] == 1)]

    if not food_positions:
      return heatmap

    if self.current_target in food_positions:
      # Target locked
      heatmap[self.current_target[0], self.current_target[1]] = 1.0
    else:
      # Uniform over all choices since it hasn't picked yet
      prob = 1.0 / len(food_positions)
      for f in food_positions:
        heatmap[f[0], f[1]] = prob
        
    return heatmap

  def select_action(self, observation, eval=False):
    my_channel = 2
    opp_channel = 3

    heatmap = self.get_subgoal_heatmap(observation)

    agent_pos_arr = np.argwhere(observation[:, :, my_channel] == 1)
    if len(agent_pos_arr) == 0:
      return np.random.randint(0, 4), None
    my_pos = tuple(agent_pos_arr[0])

    food_positions = [tuple(p) for p in np.argwhere(observation[:, :, 1] == 1)]
    if not food_positions:
      return np.random.randint(0, 4), None, heatmap

    if self.precomputed_paths is None:
      wall_pos_arr = np.argwhere(observation[:, :, 4] == 1)
      obstacles = set(tuple(p) for p in wall_pos_arr)
      self.precomputed_paths = precompute_paths(
        obstacles, observation.shape[0], observation.shape[1])

    if self.current_target not in food_positions:
      random_index = np.random.randint(0, len(food_positions))
      self.current_target = food_positions[random_index]
      self.cached_path = []

    if not self.cached_path:

      if (my_pos, self.current_target) in self.precomputed_paths:
        self.cached_path = self.precomputed_paths[(
          my_pos, self.current_target)].copy()
      else:
        # Fallback to on-the-fly A*
        wall_pos_arr = np.argwhere(observation[:, :, 4] == 1)
        obstacles = set(tuple(p) for p in wall_pos_arr)
        self.cached_path = a_star_path(
          my_pos, self.current_target, obstacles, observation.shape[0], observation.shape[1])

    if self.cached_path:
      return self.cached_path.pop(0), None, heatmap
    else:
      return np.random.randint(0, 4), None, heatmap


class GreedySwitchAgent:
  """
  An advanced opponent. It goes for the absolute closest food. 
  If it realizes the other agent is closer to that food, it abandons it and switches to another.
  """

  def __init__(self, agent_id, precomputed_paths=None):
    self.agent_id = agent_id
    self.cached_path = []
    self.current_target = None
    self.precomputed_paths = precomputed_paths

  def reset(self):
    self.cached_path = []
    self.current_target = None
  
  def get_subgoal_heatmap(self, observation):
    h, w = observation.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    agent_pos_arr = np.argwhere(observation[:, :, 2] == 1)
    opp_pos_arr = np.argwhere(observation[:, :, 3] == 1)
    food_positions = [tuple(p) for p in np.argwhere(observation[:, :, 1] == 1)]

    if len(agent_pos_arr) == 0 or len(opp_pos_arr) == 0 or not food_positions:
      return heatmap

    my_pos, opp_pos = tuple(agent_pos_arr[0]), tuple(opp_pos_arr[0])

    if self.precomputed_paths is None:
      wall_pos_arr = np.argwhere(observation[:, :, 4] == 1)
      self.precomputed_paths = precompute_paths(set(tuple(p) for p in wall_pos_arr), h, w)

    dists = []
    for f in food_positions:
      my_dist = len(self.precomputed_paths.get((my_pos, f), []))
      opp_dist = len(self.precomputed_paths.get((opp_pos, f), []))
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
      # It wants to stick with the current target, but verify safety logic
      chosen_dist = next(d for d in dists if d[2] == target_food)
      if chosen_dist[1] < chosen_dist[0]:
        safer_foods = [d for d in dists if d[0] <= d[1]]
        if safer_foods:
          safer_foods.sort(key=lambda x: x[0])
          target_food = safer_foods[0][2]
      heatmap[target_food[0], target_food[1]] = 1.0
    else:
      # Distribute probability equally among all ties, evaluating safety logic for each
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
    opp_channel = 3
    heatmap = self.get_subgoal_heatmap(observation)
    agent_pos_arr = np.argwhere(observation[:, :, my_channel] == 1)
    opp_pos_arr = np.argwhere(observation[:, :, opp_channel] == 1)

    if len(agent_pos_arr) == 0 or len(opp_pos_arr) == 0:
      return np.random.randint(0, 4), None

    my_pos = tuple(agent_pos_arr[0])
    opp_pos = tuple(opp_pos_arr[0])

    food_positions = [tuple(p) for p in np.argwhere(observation[:, :, 1] == 1)]
    if not food_positions:
      return np.random.randint(0, 4), None, heatmap

    if self.precomputed_paths is None:
      wall_pos_arr = np.argwhere(observation[:, :, 4] == 1)
      obstacles = set(tuple(p) for p in wall_pos_arr)
      self.precomputed_paths = precompute_paths(
        obstacles, observation.shape[0], observation.shape[1])

    # Compute distances to all food and sort by my distance
    dists = []
    for f in food_positions:
      my_dist = len(self.precomputed_paths.get((my_pos, f), []))
      opp_dist = len(self.precomputed_paths.get((opp_pos, f), []))
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

    # recompute path to the chosen target food only when needed
    if self.current_target != target_food or not self.cached_path:
      self.current_target = target_food
      if (my_pos, target_food) in self.precomputed_paths:
        self.cached_path = self.precomputed_paths[(my_pos, target_food)].copy()
      else:
        # Fallback to on-the-fly A*
        wall_pos_arr = np.argwhere(observation[:, :, 4] == 1)
        obstacles = set(tuple(p) for p in wall_pos_arr)
        self.cached_path = a_star_path(
          my_pos, target_food, obstacles, observation.shape[0], observation.shape[1])

    if self.cached_path:
      return self.cached_path.pop(0), None, heatmap
    else:
      return np.random.randint(0, 4), None, heatmap


class StalkerAgent:
  """
  A Hyper-Reactive Interceptor. It never chases lost races. It identifies the nearest 
  food where it has a positional advantage over the opponent, races there, and loiters 
  1 tile away to steal it at the last second.
  """

  def __init__(self, agent_id, precomputed_paths=None):
    self.agent_id = agent_id
    self.precomputed_paths = precomputed_paths

  def reset(self):
    pass

  def get_subgoal_heatmap(self, observation):
    h, w = observation.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    my_pos_arr = np.argwhere(observation[:, :, 2] == 1)
    enemy_pos_arr = np.argwhere(observation[:, :, 3] == 1)
    food_positions = [tuple(p) for p in np.argwhere(observation[:, :, 1] == 1)]

    if len(my_pos_arr) == 0 or len(enemy_pos_arr) == 0 or not food_positions:
      return heatmap

    my_pos, opp_pos = tuple(my_pos_arr[0]), tuple(enemy_pos_arr[0])

    if self.precomputed_paths is None:
      wall_pos_arr = np.argwhere(observation[:, :, 4] == 1)
      self.precomputed_paths = precompute_paths(set(tuple(p) for p in wall_pos_arr), h, w)

    winnable_foods = []
    for f in food_positions:
      e_dist = len(self.precomputed_paths.get((opp_pos, f), [])) or float('inf')
      s_dist = len(self.precomputed_paths.get((my_pos, f), [])) or float('inf')
      if s_dist <= e_dist and s_dist != float('inf'):
        winnable_foods.append((e_dist, s_dist, f))

    if winnable_foods:
      winnable_foods.sort(key=lambda x: x[0])
      min_e_dist = winnable_foods[0][0]
      tie_foods = [f for ed, sd, f in winnable_foods if ed == min_e_dist]
      
      prob = 1.0 / len(tie_foods)
      for f in tie_foods:
        heatmap[f[0], f[1]] += prob
    else:
      # Fallback to greedy distribution
      greedy_foods = []
      for f in food_positions:
        s_dist = len(self.precomputed_paths.get((my_pos, f), [])) or float('inf')
        if s_dist != float('inf'):
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
    opp_channel = 3

    heatmap = self.get_subgoal_heatmap(observation)

    my_pos_arr = np.argwhere(observation[:, :, my_channel] == 1)
    enemy_pos_arr = np.argwhere(observation[:, :, opp_channel] == 1)

    if len(my_pos_arr) == 0 or len(enemy_pos_arr) == 0:
      return np.random.randint(0, 4), None, heatmap

    my_pos = tuple(my_pos_arr[0])
    opp_pos = tuple(enemy_pos_arr[0])

    food_positions = [tuple(p) for p in np.argwhere(observation[:, :, 1] == 1)]
    if not food_positions:
      return np.random.randint(0, 4), None, heatmap

    if self.precomputed_paths is None:
      wall_pos_arr = np.argwhere(observation[:, :, 4] == 1)
      obstacles = set(tuple(p) for p in wall_pos_arr)
      self.precomputed_paths = precompute_paths(
        obstacles, observation.shape[0], observation.shape[1])

    # --- 1. Filter Unwinnable Races & Find Advantage ---
    winnable_foods = []
    for f in food_positions:
      e_path = self.precomputed_paths.get((opp_pos, f), [])
      s_path = self.precomputed_paths.get((my_pos, f), [])
      e_dist = len(e_path) if len(e_path) > 0 else float('inf')
      s_dist = len(s_path) if len(s_path) > 0 else float('inf')

      # Only target foods where we can beat or tie the opponent
      if s_dist <= e_dist and s_dist != float('inf'):
        winnable_foods.append((e_dist, s_dist, f))

    # --- 2. Select Target ---
    if winnable_foods:
      # Sort by Enemy Distance ascending. We want to intercept them as soon as possible.
      winnable_foods.sort(key=lambda x: x[0])
      min_e_dist = winnable_foods[0][0]

      # Stochastic tie-breaking
      tie_foods = [f for ed, sd, f in winnable_foods if ed == min_e_dist]
      target_food = tie_foods[np.random.randint(len(tie_foods))]

      # --- 3. AMBUSH / LOITER CHECK ---
      s_path = self.precomputed_paths.get((my_pos, target_food), [])
      s_dist = len(s_path)

      if s_dist == 1 and min_e_dist > 2:
        # We are exactly 1 tile away, and the enemy is not close enough yet.
        # LOITER: Intentionally bump a wall to skip our turn and wait.
        wall_pos_arr = np.argwhere(observation[:, :, 4] == 1)
        obstacles = set(tuple(p) for p in wall_pos_arr)

        # 0: Up, 1: Down, 2: Left, 3: Right
        for action, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
          nr, nc = my_pos[0] + dr, my_pos[1] + dc
          if (nr, nc) in obstacles:
            return action
        return np.random.randint(0, 4), None, heatmap  # Fallback

    else:
      # We are losing ALL races. The enemy has cleared their side.
      # Fallback to pure greedy to secure whatever points are left.
      greedy_foods = []
      for f in food_positions:
        s_path = self.precomputed_paths.get((my_pos, f), [])
        s_dist = len(s_path) if len(s_path) > 0 else float('inf')
        if s_dist != float('inf'):
          greedy_foods.append((s_dist, f))

      if greedy_foods:
        greedy_foods.sort(key=lambda x: x[0])
        min_s_dist = greedy_foods[0][0]
        tie_foods = [f for sd, f in greedy_foods if sd == min_s_dist]
        target_food = tie_foods[np.random.randint(len(tie_foods))]
      else:
        return np.random.randint(0, 4), None, heatmap

    # --- 4. Execution ---
    # Take a single step towards the target_food. We re-evaluate next frame.
    p_path = self.precomputed_paths.get((my_pos, target_food), [])
    if p_path:
      return p_path[0], None, heatmap
    else:
      return np.random.randint(0, 4), None, heatmap


class ChameleonAgent:
  """
  Opponent that switches between Simple and Greedy.
  """

  def __init__(self, agent_id, precomputed_paths=None):
    self.agent_id = agent_id
    self.simple_agent = SimpleAgent(agent_id, precomputed_paths)
    self.greedy_agent = GreedySwitchAgent(agent_id, precomputed_paths)
    self.current_persona = "greedy"

  def reset(self):
    self.simple_agent.reset()
    self.greedy_agent.reset()
  
  def get_subgoal_heatmap(self, observation):
    # The true prior is the weighted sum of its internal heuristic choices
    simple_hm = self.simple_agent.get_subgoal_heatmap(observation)
    greedy_hm = self.greedy_agent.get_subgoal_heatmap(observation)
    return (0.3 * simple_hm) + (0.7 * greedy_hm)

  def select_action(self, observation, eval=False):
    heatmap = self.get_subgoal_heatmap(observation)
    # 30% chance to be Simple, 70% to be Greedy
    new_persona = "simple" if np.random.rand() < 0.3 else "greedy"

    if new_persona != self.current_persona:
      self.simple_agent.reset()
      self.greedy_agent.reset()
      self.current_persona = new_persona

    if self.current_persona == "simple":
       action, _, _ =self.simple_agent.select_action(observation, eval)
    else:
       action, _, _ = self.greedy_agent.select_action(observation, eval)
    
    return action, None, heatmap

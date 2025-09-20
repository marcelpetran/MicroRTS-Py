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
        self.agents[0] = (0, self.grid_size // 2)
        self.agents[1] = (self.grid_size - 1, self.grid_size // 2)

        # Place food
        pos1 = (self.grid_size // 2, 0)
        pos2 = (self.grid_size // 2, self.grid_size - 1)
        self.food_positions.add(pos1)
        self.food_positions.add(pos2)

        self.rewards = {0: 0, 1: 0}
        self.terminal = False

        # obs, info
        return self._get_observations()
    
    def _get_action_space(self):
        return {agent_id: 4 for agent_id in range(self.num_agents)}  # 4 actions: up, down, left, right

    def _get_observations(self):
        observations = {}
        for agent_id, pos in self.agents.items():
            obs_grid = np.zeros((self.grid_size, self.grid_size, self.features), dtype=int)
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
            elif action == 3:
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
                rewards[agent_id] += 1
                self.food_positions.remove(pos)
        # obs, rewards, done, info
        return self._get_observations(), rewards, self._check_terminal(), {}
    
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
    def __init__(self, agent_id):
        self.agent_id = agent_id
        # agent randomly decides to go for top or bottom food
        self.going_for_top = True if np.random.rand() > 0.5 else False

    def reset(self):
        self.going_for_top = True if np.random.rand() > 0.5 else False

    def find_food(self, food_positions):
        if self.going_for_top:
            return min(food_positions, key=lambda x: x[0])  # food with smallest row index
        else:
            return max(food_positions, key=lambda x: x[0])  # food with largest row index

    def find_path(self, start, goal, obstacles):
        path = []
        current = start
        # also consider obstacles (other agent)
        while current != goal:
            if current[0] < goal[0] and (current[0] + 1, current[1]) not in obstacles:
                current = (current[0] + 1, current[1])
            elif current[0] > goal[0] and (current[0] - 1, current[1]) not in obstacles:
                current = (current[0] - 1, current[1])
            elif current[1] < goal[1] and (current[0], current[1] + 1) not in obstacles:
                current = (current[0], current[1] + 1)
            elif current[1] > goal[1] and (current[0], current[1] - 1) not in obstacles:
                current = (current[0], current[1] - 1)
            else:
                break  # No valid move, break to avoid infinite loop
            path.append(current)
        return path

    
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
        path = self.find_path(agent_pos, target_food)
        if path:
            return path[0]  # Take the first step in the path
        else:
            return np.random.randint(0, 4)  # Already at food position, random action
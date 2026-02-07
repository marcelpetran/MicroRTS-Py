import argparse
import logging
import time

import gymnasium as gym
import numpy as np

import lbforaging  # noqa

logger = logging.getLogger(__name__)


def _game_loop(env, render):
    """ """
    obss, _ = env.reset()
    done = False

    returns = np.zeros(env.n_agents)

    if render:
        env.render()
        time.sleep(0.5)

    while not done:
        actions = env.action_space.sample()

        obss, rewards, done, _, _ = env.step(actions)
        returns += rewards

        if render:
            env.render()
            time.sleep(0.5)

    print("Returns: ", returns)


def main(episodes=1, render=False):
    env = gym.make("Foraging-8x8-2p-3f-v3") # 8x8 grid, 2 players, 3 food, version 3
    # Set the number of agents to 2, as the env does not have this attribute by default, but in demo they use it
    env.__setattr__("n_agents", 2)  

    # OBSERVATION SPACE according to the paper [2006.07869v4] page 17 (updated to actual obs):
    # realistic observation at the example of the Foraging-8x8-2p-3f-v1 task visualised in
    # Figure 4a:
    # 
    # ( 
    #   P1 obs -- food1 --       -- food2 --       -- food3 --       -- agent1 --      -- agent2 --
    #   array([1., 4., 2.,      3., 2., 1.,         3., 5., 1.,       6., 5., 2.,       4., 4., 1.],
    #         dtype=float32),
    #   P2 obs -- food1 --       -- food2 --       -- food3 --       -- agent1 --      -- agent2 --
    #   array([1., 4., 2.,      3., 2., 1.,         3., 5., 1.,       4., 4., 1.,       6., 5., 2.],
    #         dtype=float32)
    # )
    # 
    # The observation consists of two arrays, each corresponding to the observation of one of the two agents
    # within the environment. Within that vector, triplets of the form (y, x, level) are written
    # (from top left corner starting with 0). Specifically, the first three (number of food items in the environment)
    # triplets for a total of 9 elements contain the x and y coordinates and level of each food item, 
    # and the following two (number of agents) triplets have the respective values for each agent. 
    # The coordinates always start from the ??bottom (maybe for partial obs)?? left square in the observability radius of the agent. 
    # When food items or agents are not visible, either because they are outside of the observable radius 
    # or the food has been picked up, then the respective values are replaced with (-1, -1, 0).

    # ACTION SPACE:
    # A^i = {Noop, Move North, Move South, Move West, Move East, Pickup}

    obs, _ = env.reset()
    print("Observation shape:", obs[0].shape)
    print("Action space:", env.action_space)

    print("P1 obs: ", obs[0])
    print("P2 obs: ", obs[1])
    
    if render:
        env.render()

    # for debugging / to see the obs description
    input("Press Enter to continue...")
    return

    for _ in range(episodes):
        _game_loop(env, render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--episodes", type=int, default=1, help="How many episodes to run"
    )

    args = parser.parse_args()
    main(args.episodes, args.render)
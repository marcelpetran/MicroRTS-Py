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
    env = gym.make("Foraging-8x8-2p-2f-v3")
    env.__setattr__("n_agents", 2)  # Set the number of agents to 2

    # OBSERVATION SPACE according to the environment
    # Observation = namedtuple(
    #     "Observation",
    #     ["field", "actions", "players", "game_over", "sight", "current_step"],
    # )
    # PlayerObservation = namedtuple(
    #     "PlayerObservation", ["position", "level", "history", "reward", "is_self"]
    # )  # reward is available only if is_self

    obs, _ = env.reset()
    print("Observation shape:", obs[0].shape)
    print("Action space:", env.action_space)

    print("obs: ", obs[0])

    for episode in range(episodes):
        _game_loop(env, render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--episodes", type=int, default=1, help="How many episodes to run"
    )

    args = parser.parse_args()
    main(args.episodes, args.render)
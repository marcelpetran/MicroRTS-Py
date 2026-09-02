import cProfile
import io
import pstats
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omexplore.envs.simple_foraging_env import SimpleForagingEnv
from omexplore.utils.maps import MAP_1


def run_benchmark(steps=1000):
    env = SimpleForagingEnv(max_steps=steps, map_layout=MAP_1, vision_radius=3)
    obs = env.reset()

    start_time = time.time()
    for _ in range(steps):
        actions = {0: random.choice([0, 1, 2, 3, 4]), 1: random.choice([0, 1, 2, 3, 4])}
        obs, rewards, done, info = env.step(actions)
        if done:
            obs = env.reset()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time for {steps} steps: {total_time:.4f} seconds")
    print(f"Steps per second: {steps / total_time:.2f}")


if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    run_benchmark(5000)
    pr.disable()
    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(30)
    print(s.getvalue())

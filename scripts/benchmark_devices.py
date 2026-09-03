"""Device benchmark for the team exploration pipeline.

Times, per available device (cpu / mps / cuda):
  1. env stepping (random actions, no NN)         - device-independent floor
  2. full training episodes (rollout + updates)   - the real wall-clock number
  3. isolated update() calls                     - per-training-step cost
  4. isolated select_action inference            - per-rollout-step NN cost

Run from project root:
  python scripts/benchmark_devices.py [--batch_sizes 64,128] [--episodes 3]
                                      [--updates 20] [--history_length 8]
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import wandb
from omexplore.agents.q_agent import QLearningAgent
from omexplore.agents.team_agents import TeamAgent
from omexplore.envs.roadmap_foraging_env import TeamRoadmapEnv
from omexplore.models.opponent_model import OpponentModel
from omexplore.models.transformers import SpatialOpponentModel
from omexplore.utils.omg_args import OMGArgs


def timeit(fn, repeats=1, warmup=1):
    """Median wall time of fn() over repeats, after warmup calls."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def build_agent(env, device, args_dict):
    args = OMGArgs(device=device, **args_dict)
    hostile_om = OpponentModel(SpatialOpponentModel(args), args)
    friendly_om = OpponentModel(SpatialOpponentModel(args), args)
    agent = QLearningAgent(env, hostile_om, friendly_om, args=args)
    opp = TeamAgent(env, team_id=1)
    return agent, opp


def bench_env_stepping(env, steps=400, repeats=3):
    """Env overhead with random actions (no networks involved)."""
    obs = env.reset()

    def run():
        nonlocal obs
        env.reset()
        for _ in range(steps):
            acts = {a: int(np.random.randint(8)) for a in env.agents}
            obs, _, done, _ = env.step(acts)
            if done:
                break

    return timeit(run, repeats=repeats)


def bench_select_action(agent, opp, env, repeats=50):
    """Per-step NN inference cost in rollout (both OMs + Q, batch=1)."""
    obs = env.reset()
    s_aug = agent.tracker.augment(obs[agent.learn_ids[0]])
    history = {
        "state_features": torch.zeros(
            (1, agent.args.max_history_length, agent.args.d_model),
            device=agent.device,
        ),
        "mask": torch.ones(
            (1, agent.args.max_history_length), dtype=torch.bool, device=agent.device
        ),
        "prev_obs": torch.zeros(
            (1, env.height, env.width, env.features), device=agent.device
        ),
    }

    def run():
        agent.select_action(obs[agent.learn_ids[0]], s_aug, history)

    return timeit(run, repeats=repeats)


def bench_update(agent, opp, env, n_updates=20):
    """Isolated update() cost (needs a filled replay buffer)."""
    agent.args.train_every = 1  # force updates
    ts = []
    for _ in range(n_updates):
        agent.global_step += 1
        t0 = time.perf_counter()
        agent.update()
        ts.append(time.perf_counter() - t0)
    return {"median": statistics.median(ts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="den312d")
    parser.add_argument("--batch_sizes", type=str, default="64,128")
    parser.add_argument("--episodes", type=int, default=3, help="Training episodes")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--history_length", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument(
        "--min_replay",
        type=int,
        default=500,
        help="Replay threshold before update() fires (lower = exercise update sooner)",
    )
    parser.add_argument(
        "--devices", type=str, default="cpu,mps,cuda", help="Comma-separated candidates"
    )
    args_cli = parser.parse_args()

    wandb.init(mode="disabled")
    torch.manual_seed(0)
    np.random.seed(0)

    devices = []
    for d in args_cli.devices.split(","):
        d = d.strip()
        if d == "cpu":
            devices.append(d)
        elif d == "mps" and torch.backends.mps.is_available():
            devices.append(d)
        elif d == "cuda" and torch.cuda.is_available():
            devices.append(d)
        else:
            print(f"[skip] {d}: not available")
    print(f"Devices: {devices}")
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}\n")

    for batch_size in [int(b) for b in args_cli.batch_sizes.split(",")]:
        print(f"{'=' * 60}\nBATCH SIZE {batch_size}\n{'=' * 60}")
        for device in devices:
            print(f"\n--- device: {device} (batch {batch_size}) ---")
            env = TeamRoadmapEnv(
                map_name=args_cli.map,
                max_steps=args_cli.max_steps,
                num_goals=16,
                team_sizes=(2, 2),
            )
            args_dict = dict(
                state_shape=(env.height, env.width, env.features),
                H=env.height,
                W=env.width,
                action_dim=8,
                max_steps=args_cli.max_steps,
                max_history_length=args_cli.history_length,
                capacity=5000,
                min_replay=args_cli.min_replay,
                batch_size=batch_size,
                train_every=8,
                gamma=0.995,
            )
            try:
                agent, opp = build_agent(env, device, args_dict)
            except Exception as e:
                print(f"  [FAIL] construction: {type(e).__name__}: {e}")
                continue

            # 1. env stepping (same env for all devices; measure once per device
            #    anyway - resets differ)
            t_env = bench_env_stepping(env, steps=args_cli.max_steps, repeats=2)
            print(f"  env stepping ({args_cli.max_steps} steps): {t_env:.2f}s")

            # 2. inference cost
            try:
                t_infer = bench_select_action(agent, opp, env, repeats=30)
                print(
                    f"  select_action (1 member-step): {t_infer * 1000:.1f} ms"
                    f"  -> x{len(agent.learn_ids)} members = {t_infer * len(agent.learn_ids) * 1000:.1f} ms/env-step"
                )
            except Exception as e:
                print(f"  [FAIL] select_action: {type(e).__name__}: {e}")

            # 3. training episodes (rollout + updates once replay fills)
            try:
                ep_times = []
                ep_returns = []
                for ep in range(args_cli.episodes):
                    t0 = time.perf_counter()
                    stats = agent.run_episode(opp, max_steps=args_cli.max_steps)
                    ep_times.append(time.perf_counter() - t0)
                    ep_returns.append(stats["return"])
                    print(
                        f"    ep {ep}: {ep_times[-1]:.1f}s "
                        f"(ret {stats['return']:.0f}, opp {stats['opp_return']:.0f}, "
                        f"steps {stats['steps']})"
                    )
                n_trained = len(agent.replay)
                med = statistics.median(ep_times)
                print(
                    f"  EPISODE: median {med:.1f}s, replay {n_trained}, "
                    f"returns {ep_returns}"
                )
            except Exception as e:
                print(f"  [FAIL] training episode: {type(e).__name__}: {e}")
                continue

            # 4. isolated update()
            try:
                if len(agent.replay) >= agent.args.min_replay:
                    res = bench_update(agent, opp, env, n_updates=args_cli.updates)
                    print(
                        f"  update() (B={batch_size}, L={args_cli.history_length}): "
                        f"{res['median'] * 1000:.0f} ms median"
                    )
                else:
                    print(
                        f"  [skip] update(): replay {len(agent.replay)} < "
                        f"min_replay {agent.args.min_replay}"
                    )
            except Exception as e:
                print(f"  [FAIL] update: {type(e).__name__}: {e}")

            # cleanup
            del agent, opp
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()

"""Generate a MovingAI .scen for a hand-made .map (e.g. converted MAP_4).

TeamRoadmapEnv samples agent starts AND goal positions from the scen rows
(load_scen: start=row[0], goal=row[1]), so a .map alone cannot construct
the env. This script enumerates free cells from the .map, computes pairwise
distances with the env's exact movement rules (8 directions, no corner
cutting, uniform cost 1 — see RoadmapForagingEnv._bfs), and writes
<map>-even-1.scen next to the .map.

Scen line format (see load_scen): bucket map w h startx starty goalx goaly dist
  x = column, y = row (MovingAI convention).

Usage:
    python scripts/make_scen.py --map tiny11              # write into cmapf_unified/maps
    python scripts/make_scen.py --map tiny11 --pairs 500  # subsample pairs
    python scripts/make_scen.py --selftest                # synthetic 11x11 in $TMPDIR
                                                          # + TeamRoadmapEnv smoke test
"""

import argparse
import os
import sys
import tempfile
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omexplore.envs.roadmap_foraging_env import (
    MAPS_DIR,
    TeamRoadmapEnv,
    load_movingai_map,
)

# Same move set as RoadmapForagingEnv.MOVES (8 dirs, corner cutting checked
# separately below).
MOVES = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]


def bfs(walls, height, width, source):
    """Uniform-cost BFS from source over free cells (env's exact rules)."""
    dist = np.full((height, width), -1, dtype=np.int64)
    if source in walls:
        return dist
    dist[source] = 0
    q = deque([source])
    while q:
        r, c = q.popleft()
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                if (nr, nc) not in walls and dist[nr, nc] < 0:
                    if dr and dc and ((r + dr, c) in walls or (r, c + dc) in walls):
                        continue
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
    return dist


def write_scen(map_name, height, width, pairs, out_path):
    """Write (start, goal, dist) rows as a MovingAI scen file."""
    with open(out_path, "w") as f:
        f.write("version 1\n")
        for (sr, sc), (gr, gc), d in pairs:
            # x = column, y = row.
            f.write(
                f"0\t{map_name}\t{width}\t{height}\t{sc}\t{sr}\t{gc}\t{gr}\t{d:.1f}\n"
            )
    return len(pairs)


def make_scen(map_name, maps_dir, max_pairs, rng):
    map_path = os.path.join(maps_dir, f"{map_name}.map")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"{map_path} not found")
    layout = load_movingai_map(map_path)
    height, width = len(layout), len(layout[0])
    walls = {
        (r, c) for r, row in enumerate(layout) for c, ch in enumerate(row) if ch == "#"
    }
    free = [(r, c) for r in range(height) for c in range(width) if (r, c) not in walls]
    print(
        f"map {map_name}: {height}x{width}, {len(walls)} walls, {len(free)} free cells"
    )

    # All reachable ordered pairs.
    all_pairs = []
    for s in free:
        d = bfs(walls, height, width, s)
        for g in free:
            if g != s and d[g] >= 0:
                all_pairs.append((s, g, int(d[g])))
    print(f"reachable ordered pairs: {len(all_pairs)}")

    rng.shuffle(all_pairs)
    if max_pairs and len(all_pairs) > max_pairs:
        # Keep coverage: every free cell should appear as a start AND as a
        # goal in the kept subset, then fill the rest randomly.
        kept, seen_s, seen_g, seen_pairs = [], set(), set(), set()
        for s, g, d in all_pairs:
            if s not in seen_s or g not in seen_g:
                kept.append((s, g, d))
                seen_s.add(s)
                seen_g.add(g)
                seen_pairs.add((s, g))
        for s, g, d in all_pairs:
            if len(kept) >= max_pairs:
                break
            if (s, g) not in seen_pairs:
                kept.append((s, g, d))
                seen_pairs.add((s, g))
        all_pairs = kept[:max_pairs] if len(kept) > max_pairs else kept

    out_path = os.path.join(maps_dir, f"{map_name}-even-1.scen")
    n = write_scen(map_name, height, width, all_pairs, out_path)
    print(f"wrote {n} scen rows -> {out_path}")
    return out_path


def selftest():
    """Synthetic 11x11 map from MAP_4 walls in a temp dir + env smoke test."""
    from omexplore.utils.maps import MAP_4

    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp(prefix="tiny11_")
    layout = ["".join("." if ch != "#" else "#" for ch in row) for row in MAP_4]
    height, width = len(layout), len(layout[0])
    map_path = os.path.join(tmp, "tiny11.map")
    with open(map_path, "w") as f:
        f.write("type octile\n")
        f.write(f"height {height}\n")
        f.write(f"width {width}\n")
        f.write("map\n")
        for row in layout:
            f.write(row + "\n")
    print(f"selftest map written: {map_path}")

    make_scen("tiny11", tmp, max_pairs=0, rng=rng)

    env = TeamRoadmapEnv(
        map_name="tiny11",
        max_steps=60,
        vision_radius=3,
        num_goals=6,
        team_sizes=(2, 2),
        maps_dir=tmp,
    )
    obs = env.reset()
    print(
        f"env ok: {env.height}x{env.width}, agents at "
        f"{dict(env.agents)}, {len(env.food_positions)} goals: "
        f"{sorted(env.food_positions)}"
    )
    assert obs[0].shape[:2] == (env.height, env.width)

    done, ep_ret, steps = False, 0.0, 0
    while not done and steps < env.max_steps:
        acts = {a: int(rng.integers(0, 8)) for a in env.agents}
        obs, rewards, done, info = env.step(acts)
        ep_ret += sum(rewards.values())
        steps += 1
    print(
        f"random episode: {steps} steps, total reward {ep_ret:.2f}, "
        f"food left {len(env.food_positions)}"
    )
    assert steps == env.max_steps or len(env.food_positions) == 0
    print("SELFTEST PASSED")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", type=str, default=None, help="map name in maps dir")
    p.add_argument(
        "--maps_dir",
        type=str,
        default=None,
        help="override maps dir (default: cmapf_unified/maps)",
    )
    p.add_argument(
        "--pairs",
        type=int,
        default=1500,
        help="max scen rows (0 = all reachable ordered pairs)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    if a.selftest:
        selftest()
        return
    if not a.map:
        p.error("--map is required (or use --selftest)")
    make_scen(a.map, a.maps_dir or MAPS_DIR, a.pairs, rng)


if __name__ == "__main__":
    main()

"""Roadmap graph loader for the graph-based team-exploration agent.

Loads the precomputed .roadmap files produced by
cmapf_unified/scripts/roadmap.py, snaps their continuous node coordinates
to grid cells, and recomputes everything the RL agent actually needs from
BFS distance fields.

Why recompute: the .roadmap edge weights are Euclidean line-of-sight
distances (travel time only in the _bezier variants), while our agents move
on the grid (8 directions, no corner cutting). So the file's topology is
kept but its weights are replaced with exact BFS step distances, and every
ETA feature comes from per-node BFS distance fields instead of the edges.

.roadmap format::

    N <nodes>
    <x y> per node          (x = column, y = row, continuous map units)
    W <pairs>               (optional: scenario start/goal node indices)
    <s g> per pair
    E <edges>
    <u v weight> per edge   (weight = last field on the line)

Provides:
  - dist_fields: (N, H, W) int32 exact step distance from every node,
    using the env's own movement model (MOVES from roadmap_foraging_env).
  - edge_index / edge_weight: file topology, BFS-reweighted; edges between
    mutually unreachable nodes are dropped (counted in dropped_edges).
  - region: (H, W) int16 nearest-node Voronoi assignment of free cells by
    BFS distance (for goal attachment + OM claim pooling); -1 = wall or
    not reachable from any node.
  - path_actions(): BFS action sequence from an arbitrary cell to a node
    (the low-level controller's walk).

A .bfs.npz cache is written next to the .roadmap file (invalidated by the
source file's mtime/size) so the N BFS runs happen only once per roadmap.
"""

from __future__ import annotations

import os
import re
import time
from collections import deque

import numpy as np

from omexplore.envs.roadmap_foraging_env import MAPS_DIR, MOVES, load_movingai_map

ROADMAPS_DIR = os.path.abspath(os.path.join(MAPS_DIR, "..", "viz"))

_FILENAME_RE = re.compile(r"^(?P<map>.+?)_N(?P<n>\d+)_K(?P<k>\d+)")


def list_roadmaps(map_name: str, roadmaps_dir: str = ROADMAPS_DIR) -> list[str]:
    """All .roadmap files available for a map (sorted by node count)."""
    out = []
    for fn in os.listdir(roadmaps_dir):
        if fn.endswith(".roadmap") and fn.startswith(map_name + "_"):
            out.append(os.path.join(roadmaps_dir, fn))
    return sorted(out, key=lambda p: (os.path.getsize(p), p))


def parse_roadmap(path: str):
    """Parse a .roadmap file -> (coords, waypoint_pairs, edges).

    coords: list of (x, y) floats (x = column, y = row)
    waypoint_pairs: list of (start_node, goal_node) ints (may be empty)
    edges: list of (u, v, file_weight); weight is the LAST field on the
    line, which handles both plain (`u v d`) and bezier
    (`u v cx1 cy1 cx2 cy2 tau`) edge lines.
    """
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines or not lines[0].startswith("N"):
        raise ValueError(f"{path}: expected 'N <count>' header")
    n = int(lines[0].split()[1])
    coords = []
    for l in lines[1 : 1 + n]:
        parts = l.split()
        coords.append((float(parts[0]), float(parts[1])))
    if len(coords) != n:
        raise ValueError(f"{path}: header says {n} nodes, found {len(coords)}")
    i = 1 + n
    waypoint_pairs = []
    if i < len(lines) and lines[i].startswith("W"):
        wc = int(lines[i].split()[1])
        for l in lines[i + 1 : i + 1 + wc]:
            s, g = l.split()[:2]
            waypoint_pairs.append((int(s), int(g)))
        i += 1 + wc
    if i >= len(lines) or not lines[i].startswith("E"):
        raise ValueError(f"{path}: expected 'E <count>' after nodes/W")
    ec = int(lines[i].split()[1])
    edges = []
    for l in lines[i + 1 : i + 1 + ec]:
        parts = l.split()
        edges.append((int(parts[0]), int(parts[1]), float(parts[-1])))
    bad = [e for e in edges if not (0 <= e[0] < n and 0 <= e[1] < n)]
    if bad:
        raise ValueError(f"{path}: {len(bad)} edges reference invalid node ids")
    return coords, waypoint_pairs, edges


def bfs_field(walls: set, height: int, width: int, source, prev: dict | None = None):
    """BFS distance field from source over free cells.

    Identical movement semantics to RoadmapForagingEnv._bfs: 8-direction
    MOVES, diagonal moves blocked when either orthogonal cell is a wall
    (no corner cutting). Returns an (H, W) int32 array (-1 unreachable);
    if `prev` is given it is filled with cell -> ((parent_cell), action).
    """
    dist = np.full((height, width), -1, dtype=np.int32)
    if source in walls:
        return dist
    dist[source] = 0
    q = deque([source])
    while q:
        r, c = q.popleft()
        d = dist[r, c] + 1
        for dr, dc, a in MOVES:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < height
                and 0 <= nc < width
                and dist[nr, nc] < 0
                and (nr, nc) not in walls
            ):
                if dr and dc and ((r + dr, c) in walls or (r, c + dc) in walls):
                    continue  # no corner cutting
                dist[nr, nc] = d
                if prev is not None:
                    prev[(nr, nc)] = ((r, c), a)
                q.append((nr, nc))
    return dist


def _snap_to_free(
    x: float, y: float, walls: set, height: int, width: int, exclude: set
):
    """Snap a continuous (x=col, y=row) coordinate to a free, unclaimed cell.

    Tries round/floor/ceil candidates first, then expands a BFS ring search
    through walls to the nearest free cell (also used to resolve two nodes
    snapping onto the same cell). Returns (row, col).
    """
    fx, fy = int(x), int(y)
    candidates = [
        (int(round(x)), int(round(y))),
        (fx, fy),
        (fx + 1, fy),
        (fx, fy + 1),
        (fx + 1, fy + 1),
    ]
    for cx, cy in candidates:
        if 0 <= cx < width and 0 <= cy < height:
            cell = (cy, cx)
            if cell not in walls and cell not in exclude:
                return cell
    # Ring search from the clamped rounded cell (walls are traversable here).
    sr = min(max(int(round(y)), 0), height - 1)
    sc = min(max(int(round(x)), 0), width - 1)
    start = (sr, sc)
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) not in walls and (r, c) not in exclude:
            return (r, c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
    raise RuntimeError("no free cell found during node snapping")


class RoadmapGraph:
    """Graph backbone + BFS distance machinery for one map."""

    def __init__(self, roadmap_path: str, maps_dir: str | None = None):
        self.roadmap_path = os.path.abspath(roadmap_path)
        if not os.path.exists(self.roadmap_path):
            raise FileNotFoundError(self.roadmap_path)

        stem = os.path.basename(self.roadmap_path)[: -len(".roadmap")]
        m = _FILENAME_RE.match(stem)
        self.map_name = m.group("map") if m else stem

        maps_dir = maps_dir or MAPS_DIR
        self.map_layout = load_movingai_map(
            os.path.join(maps_dir, f"{self.map_name}.map")
        )
        self.height = len(self.map_layout)
        self.width = len(self.map_layout[0])
        self.walls = {
            (r, c)
            for r, row in enumerate(self.map_layout)
            for c, ch in enumerate(row)
            if ch == "#"
        }
        self.wall_mask = np.zeros((self.height, self.width), dtype=bool)
        if self.walls:
            wr, wc = zip(*self.walls)
            mask = np.zeros((self.height, self.width), dtype=bool)
            mask[np.asarray(wr), np.asarray(wc)] = True
            self.wall_mask = mask
        self.free_cells = int((~self.wall_mask).sum())

        coords, self.waypoint_pairs, file_edges = parse_roadmap(self.roadmap_path)
        self.n_nodes = len(coords)
        self.node_xy = np.asarray(coords, dtype=np.float32)

        # --- snap nodes to free cells (resolving collisions) ---
        occupied: set = set()
        cells = []
        self.snap_displacements = []
        for x, y in coords:
            cell = _snap_to_free(x, y, self.walls, self.height, self.width, occupied)
            cells.append(cell)
            occupied.add(cell)
            self.snap_displacements.append(
                max(abs(cell[0] - int(round(y))), abs(cell[1] - int(round(x))))
            )
        self.node_cells = np.asarray(cells, dtype=np.int64)  # (N, 2) rows, cols

        # --- BFS distance fields (cached to <roadmap>.bfs.npz) ---
        cache_path = self.roadmap_path + ".bfs.npz"
        st = os.stat(self.roadmap_path)
        loaded = False
        if os.path.exists(cache_path):
            try:
                z = np.load(cache_path)
                if int(z["src_mtime"]) == int(st.st_mtime) and int(
                    z["src_size"]
                ) == int(st.st_size):
                    self.node_cells = z["node_cells"]
                    self.dist_fields = z["dist_fields"]
                    self.region = z["region"]
                    assert len(self.node_cells) == self.n_nodes
                    loaded = True
            except Exception:
                loaded = False
        if not loaded:
            t0 = time.time()
            self.dist_fields = np.stack(
                [
                    bfs_field(self.walls, self.height, self.width, tuple(cell))
                    for cell in self.node_cells
                ]
            )  # (N, H, W) int32
            self.region = self._assign_regions()
            np.savez_compressed(
                cache_path,
                src_mtime=np.int64(st.st_mtime),
                src_size=np.int64(st.st_size),
                node_cells=self.node_cells,
                dist_fields=self.dist_fields,
                region=self.region,
            )
            self._build_seconds = time.time() - t0

        # --- reweight edges with BFS distances, drop unreachable ones ---
        kept_index, kept_weight, kept_file_weight, dropped = [], [], [], 0
        for u, v, wfile in file_edges:
            ru, cu = self.node_cells[u]
            rv, cv = self.node_cells[v]
            d = int(self.dist_fields[u, rv, cv])
            d_alt = int(self.dist_fields[v, ru, cu])
            if d < 0 or d_alt < 0:
                dropped += 1
                continue
            kept_index.append((u, v))
            kept_weight.append(max(d, d_alt))
            kept_file_weight.append(wfile)
        self.edge_index = np.asarray(kept_index, dtype=np.int64).reshape(-1, 2)
        self.edge_weight = np.asarray(kept_weight, dtype=np.float32)
        self.edge_file_weight = np.asarray(kept_file_weight, dtype=np.float32)
        self.dropped_edges = dropped

        # --- region sizes ---
        self.region_size = np.bincount(
            self.region[self.region >= 0].astype(np.int64), minlength=self.n_nodes
        )

    # ------------------------------------------------------------------ #
    # Region assignment: nearest node by BFS distance (Voronoi over the
    # movement metric, not Euclidean). Ties -> lowest node id.
    # ------------------------------------------------------------------ #
    def _assign_regions(self) -> np.ndarray:
        d = self.dist_fields
        big = np.where(d >= 0, d, np.int32(1 << 30))
        region = np.argmin(big, axis=0).astype(np.int16)
        region[d.min(axis=0) < 0] = -1  # unreachable from every node
        region[self.wall_mask] = -1
        return region

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    def dist_to_nodes(self, cell) -> np.ndarray:
        """(N,) BFS step distances from an arbitrary (row, col) cell."""
        r, c = cell
        return self.dist_fields[:, r, c].copy()

    def nearest_node(self, cell) -> int:
        """Nearest node id to a cell (-1 if unreachable)."""
        r, c = cell
        if self.wall_mask[r, c]:
            return -1
        return int(self.region[r, c])

    def eta_between(self, u: int, v: int) -> int:
        """BFS steps between two nodes (-1 unreachable)."""
        r, c = self.node_cells[v]
        return int(self.dist_fields[u, r, c])

    def bfs_from(self, cell) -> np.ndarray:
        """Distance field from an arbitrary cell (env movement rules)."""
        return bfs_field(self.walls, self.height, self.width, tuple(cell))

    def path_actions(self, src_cell, node_id: int) -> list[int] | None:
        """Action sequence (env action ids) walking src_cell -> node.

        Returns [] if already at the node, None if unreachable.
        """
        target = tuple(self.node_cells[node_id])
        if tuple(src_cell) == target:
            return []
        if src_cell in self.walls:
            return None
        prev: dict = {}
        dist = bfs_field(self.walls, self.height, self.width, tuple(src_cell), prev)
        if dist[target] < 0:
            return None
        actions = []
        cur = target
        while cur != tuple(src_cell):
            parent, a = prev[cur]
            actions.append(a)
            cur = parent
        return actions[::-1]

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        """Summary used by scripts/roadmap_graph_stats.py."""
        deg = (
            np.bincount(self.edge_index.ravel(), minlength=self.n_nodes)
            if len(self.edge_index)
            else np.zeros(self.n_nodes, dtype=np.int64)
        )
        covered = (self.region >= 0) & (~self.wall_mask)
        n_free = self.free_cells
        # Farthest-in-region distance per node ("region radius").
        radii, mean_radii = [], []
        for i in range(self.n_nodes):
            mask = self.region == i
            if not mask.any():
                radii.append(-1)
                mean_radii.append(-1.0)
                continue
            vals = self.dist_fields[i][mask]
            radii.append(int(vals.max()))
            mean_radii.append(float(vals.mean()))
        # Nearest neighbouring node (BFS).
        nn = []
        for i in range(self.n_nodes):
            ds = [
                int(self.dist_fields[i, r, c])
                for j, (r, c) in enumerate(self.node_cells)
                if j != i
            ]
            ds = [d for d in ds if d >= 0]
            nn.append(min(ds) if ds else -1)
        s = {
            "map": self.map_name,
            "roadmap": os.path.basename(self.roadmap_path),
            "dims": (self.height, self.width),
            "free_cells": n_free,
            "n_nodes": self.n_nodes,
            "n_edges_file": len(self.edge_file_weight) + self.dropped_edges,
            "n_edges_kept": len(self.edge_index),
            "dropped_edges": self.dropped_edges,
            "waypoint_pairs": len(self.waypoint_pairs),
            "snap_max_displacement": int(max(self.snap_displacements))
            if self.snap_displacements
            else 0,
            "snap_moved_nodes": sum(1 for d in self.snap_displacements if d > 0),
            "degree_min": int(deg.min()),
            "degree_mean": float(deg.mean()),
            "degree_max": int(deg.max()),
        }
        if len(self.edge_weight):
            ratio = self.edge_weight / np.maximum(self.edge_file_weight, 1e-9)
            s.update(
                {
                    "weight_err_mean": float(
                        np.abs(self.edge_weight - self.edge_file_weight).mean()
                    ),
                    "weight_err_max": float(
                        np.abs(self.edge_weight - self.edge_file_weight).max()
                    ),
                    "bfs_over_file_ratio_mean": float(ratio.mean()),
                    "bfs_over_file_ratio_max": float(ratio.max()),
                }
            )
        s.update(
            {
                "region_covered_frac": float(covered.sum() / max(n_free, 1)),
                "uncovered_free_cells": int(n_free - covered.sum()),
                "region_size_min": int(self.region_size.min()),
                "region_size_mean": float(self.region_size.mean()),
                "region_size_max": int(self.region_size.max()),
                "region_radius_max": int(max(radii)),
                "region_radius_mean": float(np.mean([r for r in mean_radii if r >= 0])),
                "node_nn_dist_min": int(min(nn)),
                "node_nn_dist_mean": float(np.mean([d for d in nn if d >= 0])),
                "node_nn_dist_max": int(max(nn)),
            }
        )
        return s

    def ascii_map(self) -> str:
        """Text render: walls '#', free '.', nodes as id chars (<=62)."""
        import string

        chars = string.digits + string.ascii_lowercase + string.ascii_uppercase
        grid = [list(row) for row in self.map_layout]
        for i, (r, c) in enumerate(self.node_cells):
            grid[r][c] = chars[i] if self.n_nodes <= 62 else "o"
        return "\n".join("".join(row) for row in grid)


if __name__ == "__main__":
    import sys

    rg = RoadmapGraph(sys.argv[1])
    for k, v in rg.stats().items():
        print(f"{k:26s} {v}")

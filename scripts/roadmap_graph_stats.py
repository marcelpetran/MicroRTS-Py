"""Text-only statistics for a .roadmap graph on a MovingAI map.

Verifies, without any images, that a precomputed roadmap is usable as an
SMDP backbone for the graph-based exploration agent:

  - node / edge counts, dropped edges after BFS reweighting
  - snap displacements (continuous node coords -> free cells)
  - node degree distribution
  - file (Euclidean) vs BFS edge weights: error and worst-case ratio,
    confirming that reweighting was necessary
  - Voronoi region coverage of free cells (every cell must belong to a node)
  - region size / radius (how far a cell can be from its node)
  - nearest-node distance for every free cell
  - ASCII render of the graph over the map (small maps)

Usage:
  python scripts/roadmap_graph_stats.py cmapf_unified/viz/den312d_N199_K8.roadmap
  python scripts/roadmap_graph_stats.py cmapf_unified/viz/tiny11_N20_K4.roadmap --ascii
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from omexplore.envs.roadmap_graph import RoadmapGraph


def pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("roadmap", help="path to a .roadmap file")
    p.add_argument(
        "--maps_dir",
        default="cmapf_unified/maps",
        help="directory with <map>.map files",
    )
    p.add_argument(
        "--ascii",
        action="store_true",
        help="print ASCII map with node markers (small maps only)",
    )
    p.add_argument(
        "--rebuild", action="store_true", help="ignore the .bfs.npz cache and recompute"
    )
    a = p.parse_args()

    g = RoadmapGraph(a.roadmap, a.maps_dir)
    s = g.stats()

    w = max(len(k) for k in s)
    for k, v in s.items():
        if isinstance(v, float):
            print(f"{k:<{w}}  {v:.4f}")
        else:
            print(f"{k:<{w}}  {v}")

    # ---- derived distributions (beyond g.stats()) ----
    deg = np.bincount(g.edge_index.reshape(-1), minlength=g.n_nodes)
    print("\n-- degree percentiles --")
    for q in (5, 25, 50, 75, 95):
        print(f"  p{q:<2} {pct(deg, q):5.1f}")

    file_w, bfs_w = g.edge_file_weight, g.edge_weight
    ratio = np.where(file_w > 0, bfs_w / file_w, np.nan)
    print("\n-- edge weights (BFS / file ratio) --")
    print(
        f"  mean {np.nanmean(ratio):.3f}  min {np.nanmin(ratio):.3f}  "
        f"max {np.nanmax(ratio):.3f}"
    )
    print(
        f"  edges where BFS > file (walls in the way): "
        f"{int((ratio > 1.001).sum())}/{len(ratio)}"
    )
    print(f"  edges where BFS < file: {int((ratio < 0.999).sum())}/{len(ratio)}")

    # distance from every free cell to its nearest node
    # (dist_fields uses -1 for unreachable; mask it out before the min)
    big = np.where(g.dist_fields >= 0, g.dist_fields, 1 << 30)
    nn_d = big.min(axis=0)[~g.wall_mask]
    print("\n-- nearest node distance over free cells --")
    print(
        f"  mean {nn_d.mean():.3f}  max {nn_d.max()}  "
        f"p95 {pct(nn_d, 95):.1f}  cells with nn>8: "
        f"{int((nn_d > 8).sum())}/{len(nn_d)}"
    )

    if a.ascii:
        print("\n-- ASCII (nodes marked 0-9a-z, '.' free, '#' wall) --")
        print(g.ascii_map())


if __name__ == "__main__":
    main()

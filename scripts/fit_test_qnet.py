"""Supervised capacity test: which Q-net architecture can REPRESENT the
greedy routing policy from (obs, belief) inputs?

Context: probes on run 1xny5ufg showed the trained QNet's advantage head
cannot use the belief prior (adjacent-goal argmax below chance, flat
Q-spread). Before spending a remote training run, test the architectures
in isolation on the supervised task they must at minimum be able to fit:

    input:  obs (7ch) + belief (3ch: believed goals, opp, age) [+1 dist]
    label: first BFS action toward the NEAREST believed goal

Variants (--arch, repeatable):
  old            QNet as used in the runs (3x conv 3x3, flatten->linear heads)
  old_dist       QNet + a 4th belief channel: distance-to-nearest-believed-goal
                 field per cell, normalized (the "cheap fix" candidate)
  temporal       QNetTemporal (dilated convs, self-cell + pooled readout)
  temporal_dist  QNetTemporal + the same distance channel

If 'old' fits train but not test -> memorization without spatial
generalization. If 'temporal' fits both -> architecture unlocks the
mapping. If only *_dist fits -> the distance field is needed regardless
of architecture (conv nets struggle with min-plus/BFS-style distances).

Run:  python scripts/fit_test_qnet.py --arch old --arch temporal ...
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as Fn

import wandb  # networks reference wandb at package import
from omexplore.envs.roadmap_foraging_env import TeamRoadmapEnv
from omexplore.models.networks import QNet, QNetTemporal
from omexplore.utils.omg_args import OMGArgs

ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
    4: (-1, -1),
    5: (-1, 1),
    6: (1, -1),
    7: (1, 1),
}
INV_DELTA = {v: k for k, v in ACTION_DELTAS.items()}


def gen_sample(env, n_goals):
    """One (state, belief, dist_ch, bfs_label, straight_label) sample."""
    obs = env.reset()
    r0, c0 = env.agents[0]
    all_goals = list(env.food_positions)
    if len(all_goals) > n_goals:
        idx = np.random.choice(len(all_goals), n_goals, replace=False)
        goals = [all_goals[i] for i in idx]
    else:
        goals = all_goals

    field = env.bfs_distance_field((r0, c0))
    reachable = [(field[g], g) for g in goals if field[g] >= 0]
    if not reachable:
        return None
    _, nearest = min(reachable)
    path = env.find_path((r0, c0), nearest)
    if not path:
        return None

    belief = np.zeros((env.height, env.width, 3), dtype=np.float32)
    for g in goals:
        belief[g[0], g[1], 0] = 1.0

    # Distance-to-nearest-believed-goal field (for the old_dist variant).
    dfield = np.full((env.height, env.width), 128.0, dtype=np.float32)
    for g in goals:
        fg = env.bfs_distance_field(g)
        dfield = np.minimum(dfield, np.where(fg < 0, 128.0, fg))
    dist_ch = (1.0 - dfield / 128.0).astype(np.float32)[..., None]

    dr, dc = np.sign(nearest[0] - r0), np.sign(nearest[1] - c0)
    straight = INV_DELTA.get((int(dr), int(dc)))
    return obs[0].astype(np.float32), belief, dist_ch, path[0], straight


def build_dataset(env, n, n_goals):
    xs, bs, ds, ys = [], [], [], []
    straight_ok = 0
    while len(xs) < n:
        s = gen_sample(env, n_goals)
        if s is None:
            continue
        x, b, d, y, straight = s
        xs.append(x)
        bs.append(b)
        ds.append(d)
        ys.append(y)
        if straight is not None and straight == y:
            straight_ok += 1
    print(
        f"dataset: {len(xs)} samples | straight-line==BFS-label: "
        f"{straight_ok / len(xs):.0%} | label hist: "
        f"{np.bincount(ys, minlength=8).tolist()}"
    )
    return (
        np.stack(xs),
        np.stack(bs),
        np.stack(ds),
        np.asarray(ys, dtype=np.int64),
        straight_ok / len(xs),
    )


def make_net(arch, env, qnet_dim, cnn_hidden, belief_channels):
    args = OMGArgs(
        state_shape=(env.height, env.width, env.features),
        belief_channels=belief_channels,
        friendly_om=True,
        qnet_hidden=qnet_dim,
        cnn_hidden=cnn_hidden,
    )
    args.action_dim = 8
    cls = QNetTemporal if arch.startswith("temporal") else QNet
    return cls(args)


@torch.no_grad()
def accuracy(net, x, b, d, y, use_dist, device, g0, chunk=128):
    correct = 0
    for i in range(0, len(y), chunk):
        xb = torch.from_numpy(x[i : i + chunk]).to(device)
        bb = torch.from_numpy(b[i : i + chunk]).to(device)
        db = torch.from_numpy(d[i : i + chunk]).to(device)
        s_aug = (
            torch.cat([xb, bb, db], dim=-1) if use_dist else torch.cat([xb, bb], dim=-1)
        )
        gg = g0.expand(len(s_aug), -1, -1)
        pred = net(s_aug, gg, gg).argmax(dim=1).cpu().numpy()
        correct += (pred == y[i : i + chunk]).sum()
    return correct / len(y)


def run_variant(arch, data, a, device):
    xtr, btr, dtr, ytr, *_ = data["train"]
    xte, bte, dte, yte, *_ = data["test"]
    use_dist = arch.endswith("_dist")
    n_ch = 11 if use_dist else 10

    net = make_net(
        arch, data["env"], a.qnet_dim, a.cnn_hidden, 4 if use_dist else 3
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"\n--- arch={arch} ({n_ch} input ch, {n_params / 1e6:.1f}M params) ---")

    g0 = torch.zeros((1, data["env"].height, data["env"].width), device=device)
    xtr_t = torch.from_numpy(xtr).to(device)
    btr_t = torch.from_numpy(btr).to(device)
    dtr_t = torch.from_numpy(dtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    n = len(ytr_t)
    best_test = 0.0
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, n, (a.batch,), device=device)
        s = torch.cat([xtr_t[idx], btr_t[idx]], dim=-1)
        if use_dist:
            s = torch.cat([s, dtr_t[idx]], dim=-1)
        gg = g0.expand(a.batch, -1, -1)
        logits = net(s, gg, gg)
        loss = Fn.cross_entropy(logits, ytr_t[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % a.eval_every == 0 or step == a.steps:
            tr = accuracy(
                net, xtr[:512], btr[:512], dtr[:512], ytr[:512], use_dist, device, g0
            )
            te = accuracy(net, xte, bte, dte, yte, use_dist, device, g0)
            best_test = max(best_test, te)
            print(
                f"step {step:5d} | loss {loss.item():.4f} | "
                f"train acc {tr:.1%} | test acc {te:.1%}"
            )
    return {"arch": arch, "params": n_params, "best_test": best_test, "final_test": te}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arch",
        action="append",
        choices=["old", "old_dist", "temporal", "temporal_dist"],
        default=None,
    )
    p.add_argument("--map", type=str, default="den312d")
    p.add_argument("--num_goals", type=int, default=64)
    p.add_argument("--n_goals", type=int, default=64, help="believed goals per sample")
    p.add_argument("--vision_radius", type=int, default=10)
    p.add_argument("--team_sizes", type=str, default="2,2")
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--n_train", type=int, default=1600)
    p.add_argument("--n_test", type=int, default=400)
    p.add_argument("--qnet_dim", type=int, default=512)
    p.add_argument("--cnn_hidden", type=int, default=64)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval_every", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if not a.arch:
        a.arch = ["old", "old_dist", "temporal", "temporal_dist"]

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    wandb.init(mode="disabled")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    env = TeamRoadmapEnv(
        map_name=a.map,
        max_steps=a.max_steps,
        vision_radius=a.vision_radius,
        num_goals=a.num_goals,
        team_sizes=tuple(int(x) for x in a.team_sizes.split(",")),
    )
    print("building datasets (BFS fields cached on first pass)...")
    data = {
        "env": env,
        "train": build_dataset(env, a.n_train, a.n_goals),
        "test": build_dataset(env, a.n_test, a.n_goals),
    }

    results = []
    for arch in a.arch:
        results.append(run_variant(arch, data, a, device))

    print("\n================ SUMMARY ================")
    straight = data["train"][4]
    print(f"straight-line baseline      : {straight:.1%}")
    for r in results:
        print(
            f"{r['arch']:<10} ({r['params'] / 1e6:7.1f}M params): "
            f"best test acc {r['best_test']:.1%}"
        )
    print("=========================================")


if __name__ == "__main__":
    main()

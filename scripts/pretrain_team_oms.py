"""Pretrain the hostile + friendly OMs on scripted team data.

Consumes the dataset from omexplore.collect_data (team mode):
  python -m omexplore.collect_data team --episodes 25

Then (from project root):
  /opt/homebrew/anaconda3/envs/om/bin/python scripts/pretrain_team_oms.py \
      --dataset ./dataset/team_dataset.pt --out_dir ./models/pretrained_oms

The dataset items are bit-packed (state, history) + sparse labels:
  - hostile OM trains on  true_goal_cells         (hindsight claims)
                        + true_opp_heatmap_cells  (decision-time intent)
  - friendly OM trains on true_team_cells         (teammates' claims,
                        excluding the acting agent)
                        + true_team_heatmap_cells (teammates' intent)

Architecture flags must match scripts/train_team_exploration.py defaults
(d_model, nhead, ...) or the checkpoints will not load there.
Afterwards, warm-start training with:
  python scripts/train_team_exploration.py --pretrained_om ./models/pretrained_oms
"""

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

import wandb
from omexplore.models.opponent_model import OpponentModel
from omexplore.models.transformers import SpatialOpponentModel
from omexplore.utils.labeling import dense_from_sparse
from omexplore.utils.omg_args import OMGArgs
from omexplore.utils.packing import unpack_obs

torch.set_float32_matmul_precision("high")


def collate_history_packed(items, args, device):
    """Packed-aware twin of OpponentModel.collate_history (extra=0)."""
    max_len = args.max_history_length
    B = len(items)
    H, W, F = args.state_shape
    states = torch.zeros((B, max_len, H, W, F), dtype=torch.float32, device=device)
    mask = torch.zeros((B, max_len), dtype=torch.bool, device=device)
    prev_first = torch.zeros((B, H, W, F), dtype=torch.float32, device=device)
    for i, t in enumerate(items):
        seq = t["history"]["states"]  # (T, H, W, 1) packed uint8
        end = min(t["hist_len"], len(seq))
        L = min(end, max_len)
        if L <= 0:
            continue
        start = end - L
        window = np.unpackbits(seq[start:end], axis=-1, count=F)  # (L, H, W, F)
        states[i, -L:] = torch.from_numpy(window).float()
        mask[i, -L:] = True
        if start > 0:
            pf = np.unpackbits(seq[start - 1 : start], axis=-1, count=F)[0]
            prev_first[i] = torch.from_numpy(pf).float()
    return {"states": states, "mask": mask, "prev_first": prev_first}


def build_om_batch(items, args, device, goal_key, heatmap_key):
    """Unpack a batch and densify the sparse labels for pretrain_step."""
    H, W = args.state_shape[0], args.state_shape[1]
    F = args.state_shape[2]
    states_packed = np.stack([b["state"] for b in items])  # (B, H, W, 1)
    states = np.unpackbits(states_packed, axis=3, count=F)
    return {
        "states": torch.from_numpy(states).float().to(device),
        "history": collate_history_packed(items, args, device),
        "true_goal_map": torch.from_numpy(
            dense_from_sparse([b[goal_key] for b in items], H, W)
        ).to(device),
        "true_opp_heatmap": torch.from_numpy(
            dense_from_sparse([b[heatmap_key] for b in items], H, W)
        ).to(device),
    }


def pretrain_oms(dataset, args, epochs, batch_size, device, use_wandb=True):
    """Pretrain hostile + friendly OMs; returns the two OpponentModels."""
    hostile_om = OpponentModel(SpatialOpponentModel(args), args)
    friendly_om = OpponentModel(SpatialOpponentModel(args), args)

    for epoch in range(epochs):
        order = list(range(len(dataset)))
        random.shuffle(order)
        h_losses, f_losses, h_kls, f_kls = [], [], [], []
        pbar = tqdm(
            range(0, len(order), batch_size), desc=f"Epoch {epoch + 1}/{epochs}"
        )
        for i in pbar:
            items = [dataset[j] for j in order[i : i + batch_size]]

            h_batch = build_om_batch(
                items, args, device, "true_goal_cells", "true_opp_heatmap_cells"
            )
            loss, kl, _ = hostile_om.pretrain_step(h_batch)
            h_losses.append(loss)
            h_kls.append(kl)

            f_batch = build_om_batch(
                items, args, device, "true_team_cells", "true_team_heatmap_cells"
            )
            loss, kl, _ = friendly_om.pretrain_step(f_batch)
            f_losses.append(loss)
            f_kls.append(kl)

            pbar.set_postfix(
                h_loss=f"{np.mean(h_losses[-20:]):.4f}",
                f_loss=f"{np.mean(f_losses[-20:]):.4f}",
            )

        metrics = {
            "pretrain/hostile_loss": float(np.mean(h_losses)),
            "pretrain/friendly_loss": float(np.mean(f_losses)),
            "pretrain/hostile_kl": float(np.mean(h_kls)),
            "pretrain/friendly_kl": float(np.mean(f_kls)),
            "epoch": epoch + 1,
        }
        if use_wandb:
            wandb.log(metrics)
        print(
            f"Epoch {epoch + 1:02d} | hostile {metrics['pretrain/hostile_loss']:.4f} "
            f"(kl {metrics['pretrain/hostile_kl']:.4f}) | "
            f"friendly {metrics['pretrain/friendly_loss']:.4f} "
            f"(kl {metrics['pretrain/friendly_kl']:.4f})"
        )

    return hostile_om, friendly_om


def main():
    parser = argparse.ArgumentParser(description="Team OM pretraining")
    parser.add_argument("--dataset", type=str, default="./dataset/team_dataset.pt")
    parser.add_argument("--out_dir", type=str, default="./models/pretrained_oms")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_om", type=float, default=1e-4)
    # Architecture flags — keep in sync with train_team_exploration.py.
    parser.add_argument("--max_history_length", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_encoder_layers", type=int, default=1)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cnn_hidden", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default="om-team-pretrain")
    parser.add_argument("--no_wandb", action="store_true")
    args_parsed = parser.parse_args()

    if args_parsed.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args_parsed.device

    random.seed(args_parsed.seed)
    np.random.seed(args_parsed.seed)
    torch.manual_seed(args_parsed.seed)

    print(f"Loading dataset from {args_parsed.dataset} ...")
    dataset = torch.load(args_parsed.dataset, weights_only=False)
    print(f"  {len(dataset)} transitions on device {device}")

    sample = unpack_obs(dataset[0]["state"])
    H, W, F = sample.shape
    print(f"  state shape {H}x{W}x{F}")

    args = OMGArgs(
        device=device,
        batch_size=args_parsed.batch_size,
        lr_om=args_parsed.lr_om,
        state_shape=(H, W, F),
        H=H,
        W=W,
        max_history_length=args_parsed.max_history_length,
        d_model=args_parsed.d_model,
        nhead=args_parsed.nhead,
        num_encoder_layers=args_parsed.num_encoder_layers,
        dim_feedforward=args_parsed.dim_feedforward,
        dropout=args_parsed.dropout,
        cnn_hidden=args_parsed.cnn_hidden,
    )

    use_wandb = not args_parsed.no_wandb
    if use_wandb:
        wandb.init(
            project=args_parsed.wandb_project,
            config=vars(args_parsed),
            name=f"pretrain_{os.path.basename(args_parsed.dataset)}",
        )
    else:
        wandb.init(mode="disabled")

    hostile_om, friendly_om = pretrain_oms(
        dataset,
        args,
        args_parsed.epochs,
        args_parsed.batch_size,
        device,
        use_wandb=use_wandb,
    )

    os.makedirs(args_parsed.out_dir, exist_ok=True)
    torch.save(
        hostile_om.inference_model.state_dict(), f"{args_parsed.out_dir}/hostile_om.pth"
    )
    torch.save(
        friendly_om.inference_model.state_dict(),
        f"{args_parsed.out_dir}/friendly_om.pth",
    )
    print(f"Saved pretrained OMs to {args_parsed.out_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()

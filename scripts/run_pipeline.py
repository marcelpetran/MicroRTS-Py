#!/usr/bin/env python3
"""Single-config driver for the team-exploration pipeline.

One key=value config file (default ./pipeline.args, untracked) controls
all three stages, so OM architecture flags set once are forwarded to BOTH
scripts/pretrain_team_oms.py and scripts/train_team_exploration.py (the
flags that must stay in sync), and env settings flow to collection and
training alike. The stage scripts themselves are untouched.

Stages:
  collect    python -m omexplore.collect_data team     (scripted 2v2 data)
  pretrain   python scripts/pretrain_team_oms.py       (hostile+friendly OMs)
  train      python scripts/train_team_exploration.py  (warm-started RL)

Usage:
  python scripts/run_pipeline.py --config ./pipeline.args \
      [--folder_id ID] [--stages collect,pretrain,train] [--force] [--dry_run]

folder_id defaults to $SLURM_JOB_ID (else 0) and names ./runs/<id>/, where
stage outputs are kept. Stages whose outputs already exist are skipped
(--force overrides):
  ./runs/<id>/team_dataset.pt
  ./runs/<id>/pretrained_oms/{hostile,friendly}_om.pth
The config is archived to ./runs/<id>/pipeline.args for provenance, so each
run directory records exactly the flags it was launched with.

Config keys are validated against the table below; a typo is a hard error,
not a silently dropped flag. seed=auto (the default when unset) derives the
seed from the SLURM job id, so every run differs without manual editing.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# config key -> (cli flag, stages it is forwarded to)
SPEC = {
    # env / shared
    "map": ("--map", ("collect", "train")),
    "team_sizes": ("--team_sizes", ("collect", "train")),
    "num_goals": ("--num_goals", ("collect", "train")),
    "vision_radius": ("--vision_radius", ("collect", "train")),
    "max_steps": ("--max_steps", ("collect", "train")),
    "seed": ("--seed", ("collect", "pretrain", "train")),
    # OM architecture — must match between pretrain and train
    "d_model": ("--d_model", ("pretrain", "train")),
    "nhead": ("--nhead", ("pretrain", "train")),
    "num_encoder_layers": ("--num_encoder_layers", ("pretrain", "train")),
    "dim_feedforward": ("--dim_feedforward", ("pretrain", "train")),
    "dropout": ("--dropout", ("pretrain", "train")),
    "cnn_hidden": ("--cnn_hidden", ("pretrain", "train")),
    "max_history_length": ("--max_history_length", ("pretrain", "train")),
    # collection
    "collect_episodes": ("--episodes", ("collect",)),
    # OM pretraining
    "pretrain_epochs": ("--epochs", ("pretrain",)),
    "pretrain_batch_size": ("--batch_size", ("pretrain",)),
    "lr_om": ("--lr_om", ("pretrain",)),
    "pretrain_wandb_project": ("--wandb_project", ("pretrain",)),
    # RL training
    "episodes": ("--episodes", ("train",)),
    "episodes_per_epoch": ("--episodes_per_epoch", ("train",)),
    "eval_episodes": ("--eval_episodes", ("train",)),
    "batch_size": ("--batch_size", ("train",)),
    "replay_capacity": ("--replay_capacity", ("train",)),
    "min_replay": ("--min_replay", ("train",)),
    "train_every": ("--train_every", ("train",)),
    "gamma": ("--gamma", ("train",)),
    "qnet_dim": ("--qnet_dim", ("train",)),
    "tau_start": ("--tau_start", ("train",)),
    "tau_end": ("--tau_end", ("train",)),
    "tau_decay_steps": ("--tau_decay_steps", ("train",)),
    "wandb_project": ("--wandb_project", ("train",)),
    # devices
    "device": ("--device", ("pretrain", "train")),
}

# Boolean keys: "key=true|false" (flag carries no value on the CLI).
BOOL_KEYS = {
    "friendly_om": ("train",),  # false -> --no_friendly_om (ablation)
    "no_wandb": ("pretrain", "train"),
}

ALL_STAGES = ("collect", "pretrain", "train")


def die(msg):
    print(f"run_pipeline: ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def load_config(path: Path) -> dict:
    cfg = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            die(f"{path}:{lineno}: expected key=value, got: {raw!r}")
        key, val = (part.strip() for part in line.split("=", 1))
        if key not in SPEC and key not in BOOL_KEYS:
            known = "\n  ".join(sorted({*SPEC, *BOOL_KEYS}))
            die(f"{path}:{lineno}: unknown key '{key}'. Valid keys:\n  {known}")
        cfg[key] = val
    return cfg


def _as_bool(val):
    v = val.strip().lower()
    if v in ("1", "true", "yes"):
        return True
    if v in ("0", "false", "no"):
        return False
    die(f"expected true/false, got {val!r}")


def derive_seed(folder_id) -> int:
    """seed=auto: use the SLURM job id (folder_id) so every run differs."""
    s = str(folder_id)
    if s.isdigit():
        return int(s)
    # named folder ids get a stable hash (str hash is process-randomized)
    return zlib.crc32(s.encode()) % (2**31)


def build_cmd(stage, cfg, dataset: Path, om_dir: Path, folder_id) -> list:
    if stage == "collect":
        cmd = [
            sys.executable,
            "-m",
            "omexplore.collect_data",
            "team",
            "--save_path",
            str(dataset),
        ]
    elif stage == "pretrain":
        cmd = [
            sys.executable,
            "scripts/pretrain_team_oms.py",
            "--dataset",
            str(dataset),
            "--out_dir",
            str(om_dir),
        ]
    else:
        cmd = [
            sys.executable,
            "scripts/train_team_exploration.py",
            "--folder_id",
            str(folder_id),
        ]
        if (om_dir / "hostile_om.pth").exists() and (
            om_dir / "friendly_om.pth"
        ).exists():
            cmd += ["--pretrained_om", str(om_dir)]

    for key, val in cfg.items():
        if key in BOOL_KEYS:
            if stage not in BOOL_KEYS[key]:
                continue
            b = _as_bool(val)
            if key == "friendly_om":
                cmd += ["--friendly_om" if b else "--no_friendly_om"]
            elif b:  # no_wandb
                cmd += ["--no_wandb"]
            continue
        flag, stages = SPEC[key]
        if stage in stages:
            cmd += [flag, val]
    return cmd


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="./pipeline.args")
    ap.add_argument(
        "--folder_id",
        default=os.environ.get("SLURM_JOB_ID", "0"),
        help="run id; defaults to $SLURM_JOB_ID (else 0)",
    )
    ap.add_argument(
        "--stages",
        default=",".join(ALL_STAGES),
        help="comma-separated subset of: collect,pretrain,train",
    )
    ap.add_argument(
        "--force", action="store_true", help="rerun stages whose outputs already exist"
    )
    ap.add_argument("--dry_run", action="store_true", help="print commands only")
    a = ap.parse_args()

    cfg_path = Path(a.config).resolve()
    if not cfg_path.exists():
        die(f"config file not found: {cfg_path}")
    cfg = load_config(cfg_path)

    # seed=auto (or unset) -> derive from the SLURM job id, so multiple runs
    # (main arm, ablation arm, replicas) differ without manual editing.
    seed_val = cfg.get("seed", "auto").strip().lower()
    if seed_val == "auto":
        cfg["seed"] = str(derive_seed(a.folder_id))
        print(f"seed   : auto -> {cfg['seed']} (from folder_id / $SLURM_JOB_ID)")

    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    for s in stages:
        if s not in ALL_STAGES:
            die(f"unknown stage '{s}' (valid: {', '.join(ALL_STAGES)})")

    run_dir = REPO / "runs" / str(a.folder_id)
    dataset = run_dir / "team_dataset.pt"
    om_dir = run_dir / "pretrained_oms"

    print(f"config : {cfg_path}")
    for k in sorted(cfg):
        print(f"  {k} = {cfg[k]}")
    print(f"run dir: {run_dir}")

    if not a.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfg_path, run_dir / "pipeline.args")

    for stage in stages:
        if stage == "collect" and dataset.exists() and not a.force:
            print(f"\n=== [{stage}] skipped ({dataset.name} exists; use --force)")
            continue
        if stage == "pretrain" and not a.dry_run:
            if not dataset.exists():
                die(f"pretrain needs {dataset} — run the collect stage first")
            if (
                (om_dir / "hostile_om.pth").exists()
                and (om_dir / "friendly_om.pth").exists()
                and not a.force
            ):
                print("\n=== [pretrain] skipped (pretrained_oms exists; use --force)")
                continue
        if stage == "train" and not a.dry_run:
            has_oms = (om_dir / "hostile_om.pth").exists() and (
                om_dir / "friendly_om.pth"
            ).exists()
            if not has_oms:
                print("WARNING: no pretrained OMs found — training from scratch")

        cmd = build_cmd(stage, cfg, dataset, om_dir, a.folder_id)
        print(f"\n=== [{stage}]\n{' '.join(cmd)}")
        if a.dry_run:
            continue
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=REPO).returncode
        if rc != 0:
            die(f"stage '{stage}' failed with exit code {rc}")
        print(f"=== [{stage}] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

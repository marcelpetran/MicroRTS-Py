import math
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import wandb
from omexplore.models.transformers import SpatialOpponentModel
from omexplore.utils.omg_args import OMGArgs


class OpponentModel(nn.Module):
    """Trains a SpatialOpponentModel on hindsight claim maps.

    inference_model is the online network used for acting and training;
    tgt_model is a frozen copy that is soft-updated each step and provides
    stable targets for the Q-learning agent.
    """

    def __init__(self, model: SpatialOpponentModel, args: OMGArgs = OMGArgs()):
        super().__init__()
        self.inference_model = model.to(args.device)
        self.tgt_model = SpatialOpponentModel(args).to(args.device)
        self.tgt_model.load_state_dict(self.inference_model.state_dict())
        for param in self.tgt_model.parameters():
            param.requires_grad = False
        self.tgt_model.eval()
        self.optimizer = torch.optim.Adam(
            self.inference_model.parameters(), lr=args.lr_om
        )
        self.device = args.device
        self.args = args

    def _sigma(self) -> float:
        """Current sigma of the label-blur schedule (linear decay)."""
        if self.args.sigma_decay_steps <= 0:
            return self.args.sigma_end
        state = self.optimizer.state_dict().get("state", [])
        step = min(
            state[0]["step"] if len(state) > 0 and "step" in state[0] else 0,
            self.args.sigma_decay_steps,
        )
        sigma = self.args.sigma - (self.args.sigma - self.args.sigma_end) * (
            step / self.args.sigma_decay_steps
        )
        return max(sigma, self.args.sigma_end)

    def collate_history(self, items, extra: int = 0) -> Dict[str, torch.Tensor]:
        max_len = self.args.max_history_length
        B = len(items)
        H, W, F_dim = self.args.state_shape
        states = torch.zeros(
            (B, max_len, H, W, F_dim), dtype=torch.float32, device=self.device
        )
        mask = torch.zeros((B, max_len), dtype=torch.bool, device=self.device)
        # True predecessor observation of each window's oldest frame (zeros at
        # episode start), so training recomputes exactly the features the
        # rollout used (see SpatialOpponentModel.forward).
        prev_first = torch.zeros(
            (B, H, W, F_dim), dtype=torch.float32, device=self.device
        )
        for i, t in enumerate(items):
            seq = t["history"]["states"]
            if not isinstance(seq, np.ndarray):
                seq = np.stack(seq)
            end = min(t["hist_len"] + extra, len(seq))
            L = min(end, max_len)
            if L <= 0:
                continue
            start = end - L
            states[i, -L:] = torch.from_numpy(seq[start:end]).float()
            mask[i, -L:] = True
            if start > 0:
                prev_first[i] = torch.from_numpy(seq[start - 1]).float()
        return {"states": states, "mask": mask, "prev_first": prev_first}

    def collate_cached_history_pair(
        self, items, key: str
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Collate the per-step cached OM features into current + next windows.

        Matches the rollout's rolling-feature buffer exactly: transition t
        (hist_len L, state = episode index L, next_state = index L+1) had, at
        decision time, a feature window over states (L-T..L-1) right-aligned
        into a T-slot buffer + prev_obs = the raw anchor state L-1.

        Returns two dicts, each {"state_features": (B,T,d_model),
        "mask": (B,T), "prev_obs": (B,H,W,F)}:
          cur  -> OM history for the current state s (state L)
          nxt  -> OM history for the next state s' (state L+1): window states
                  (L-T+1..L) + prev_obs = anchor state L.

        key selects which OM's cached features to use ("feats_hostile" /
        "feats_friendly"); features and states are the ANCHOR stream shared by
        reference across each episode's transitions. Unlike collate_history,
        this never re-embeds raw states, so the update path skips the CNN.
        """
        max_len = self.args.max_history_length
        B = len(items)
        H, W, F_dim = self.args.state_shape
        dmodel = self.args.d_model

        cur_feats = torch.zeros(
            (B, max_len, dmodel), dtype=torch.float32, device=self.device
        )
        cur_mask = torch.zeros((B, max_len), dtype=torch.bool, device=self.device)
        nxt_feats = torch.zeros(
            (B, max_len, dmodel), dtype=torch.float32, device=self.device
        )
        nxt_mask = torch.zeros((B, max_len), dtype=torch.bool, device=self.device)
        cur_prev = torch.zeros(
            (B, H, W, F_dim), dtype=torch.float32, device=self.device
        )
        nxt_prev = torch.zeros(
            (B, H, W, F_dim), dtype=torch.float32, device=self.device
        )

        for i, t in enumerate(items):
            feats = t["history"][key]
            states = t["history"]["states"]
            L = t["hist_len"]
            if not isinstance(feats, np.ndarray):
                feats = np.stack(feats)
            if not isinstance(states, np.ndarray):
                states = np.stack(states)

            # current window: states (L-take .. L-1), right-aligned
            take = min(L, max_len)
            if take > 0:
                cur_feats[i, -take:] = torch.from_numpy(
                    feats[L - take : L].astype(np.float32)
                )
                cur_mask[i, -take:] = True
                cur_prev[i] = torch.from_numpy(states[L - 1].astype(np.float32))

            # next window: states (L+1-take_n .. L), right-aligned
            take_n = min(L + 1, max_len)
            if take_n > 0:
                nxt_feats[i, -take_n:] = torch.from_numpy(
                    feats[L + 1 - take_n : L + 1].astype(np.float32)
                )
                nxt_mask[i, -take_n:] = True
                nxt_prev[i] = torch.from_numpy(states[L].astype(np.float32))

        return {
            "cur": {
                "state_features": cur_feats,
                "mask": cur_mask,
                "prev_obs": cur_prev,
            },
            "nxt": {
                "state_features": nxt_feats,
                "mask": nxt_mask,
                "prev_obs": nxt_prev,
            },
        }

    def heatmap_kl_divergence(
        self, g_map: torch.Tensor, true_goal_map: torch.Tensor
    ) -> float:
        """
        Evaluates how closely the inferred subgoal distribution matches the true intent distribution
        using Kullback-Leibler Divergence. Lower is better (0.0 is perfect).

        Args:
            g_map (torch.Tensor): Inferred subgoal heatmap (softmaxed), shape (B, H, W)
            true_goal_map (torch.Tensor): Ground truth distribution over subgoals, shape (B, H, W)
        """
        B = g_map.shape[0]
        g_map_flat = g_map.view(B, -1)  # (B, H*W)
        true_goal_flat = true_goal_map.view(B, -1)  # (B, H*W)
        log_g_map = torch.log(g_map_flat + 1e-8)
        kl_div = F.kl_div(log_g_map, true_goal_flat, reduction="batchmean")

        return kl_div.item()

    def _pairwise_manhattan(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Cached pairwise Manhattan distance matrix of shape (H*W, H*W) for the grid.
        Built lazily and keyed by (H, W, device) so it is reused across calls.
        """
        key = (H, W, device)
        cache = getattr(self, "_pw_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        coords = torch.stack([y, x], dim=-1).reshape(-1, 2).float()  # (HW, 2)
        dist = (coords.unsqueeze(1) - coords.unsqueeze(0)).abs().sum(-1)  # (HW, HW)
        self._pw_cache = (key, dist)
        return dist

    def heatmap_mae(self, g_map: torch.Tensor, true_goal_map: torch.Tensor) -> float:
        """
        Mean absolute error between the predicted per-cell target
        probabilities and the binarized claim map (any claim -> 1). Lower is
        better (0.0 is perfect).

        g_map: sigmoid probabilities (B, H, W)
        true_goal_map: claim counts (B, H, W); binarized here
        """
        tgt = (true_goal_map > 0).float()
        return (g_map - tgt).abs().mean().item()

    def expected_spatial_error(
        self, g_map: torch.Tensor, true_goal_map: torch.Tensor
    ) -> float:
        """
        Probability-weighted Manhattan distance from each cell to the nearest
        valid target, averaged over the batch. Vectorized over the batch
        dimension.

        g_map: predicted per-cell probabilities (B, H, W); normalized
        internally (per-cell maps need not sum to 1)
        true_goal_map: ground-truth claim map; cells > 0 count as targets
        """
        B, H, W = g_map.shape
        dist = self._pairwise_manhattan(H, W, g_map.device)  # (HW, HW)

        g_flat = g_map.reshape(B, -1)  # (B, HW)
        g_flat = g_flat / g_flat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        tgt_flat = true_goal_map.reshape(B, -1)  # (B, HW)
        valid = tgt_flat > 0  # (B, HW)

        # Per (batch, cell): Manhattan distance to the nearest valid target.
        # Invalid targets get a sentinel larger than any real distance (max is H+W-2).
        nearest = (
            dist.unsqueeze(0)
            .masked_fill(~valid.unsqueeze(1), float(H + W))
            .amin(dim=-1)
        )  # (B, HW)

        has_target = valid.any(dim=-1)  # (B,)
        per_batch = (g_flat * nearest).sum(dim=-1)
        per_batch = torch.where(has_target, per_batch, torch.zeros_like(per_batch))

        valid_count = has_target.sum()
        if valid_count.item() == 0:
            return 0.0
        return (per_batch.sum() / valid_count).item()

    def pretrain(self, dataset, epochs=10, batch_size=128):
        """
        Pretraining loop with progress bars, throttled step-level logging and
        per-epoch summaries.
        """
        print(f"Starting pretraining for {epochs} epochs on {self.device}...")
        global_step = 0
        for epoch in range(epochs):
            random.shuffle(dataset)
            epoch_losses = []
            epoch_maes = []
            epoch_spatial_errors = []

            pbar = tqdm(
                range(0, len(dataset), batch_size), desc=f"Epoch {epoch + 1}/{epochs}"
            )

            for i in pbar:
                batch_data = dataset[i : i + batch_size]
                om_batch = {
                    "states": torch.from_numpy(
                        np.stack([b["state"] for b in batch_data], dtype=np.float32)
                    ).to(self.device, non_blocking=True),
                    "history": self.collate_history(batch_data),
                    "true_goal_map": torch.from_numpy(
                        np.stack(
                            [b["true_goal_map"] for b in batch_data], dtype=np.float32
                        )
                    ).to(self.device, non_blocking=True),
                }

                loss, mae_error, spatial_error = self.pretrain_step(om_batch)
                epoch_losses.append(loss)
                epoch_maes.append(mae_error)
                epoch_spatial_errors.append(spatial_error)
                global_step += 1
                pbar.set_postfix({"loss": f"{loss:.4f}"})

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_mae = sum(epoch_maes) / len(epoch_maes)
            avg_spatial_error = sum(epoch_spatial_errors) / len(epoch_spatial_errors)
            print(f"  => Average Loss: {avg_loss:.6f}")

            wandb.log(
                {
                    "pretrain/epoch_loss": avg_loss,
                    "pretrain/epoch_target_mae": avg_mae,
                    "pretrain/epoch_spatial_error": avg_spatial_error,
                    "epoch": epoch,
                }
            )

    def forward(
        self, x: torch.Tensor, history: Dict, cached_features=True
    ) -> torch.Tensor:
        """Predict the subgoal heatmap (B, H, W) for state x (B, H, W, F)."""
        self.inference_model.eval()
        return self.inference_model(x, history, cached_features=cached_features)

    def _bce_loss(self, pred_logits: torch.Tensor, soft_targets: torch.Tensor):
        """Class-balanced BCE for per-cell "is this a target" probabilities.

        Positive cells are extremely rare (~2 of H*W), so the positive term
        is weighted by neg_mass/pos_mass derived from the soft targets —
        without it the model collapses to all-zeros.
        """
        flat_t = soft_targets.view(soft_targets.shape[0], -1)
        pos_mass = flat_t.sum()
        pos_weight = ((flat_t.numel() - pos_mass) / pos_mass.clamp_min(1.0)).clamp_min(
            1.0
        )
        return F.binary_cross_entropy_with_logits(
            pred_logits.view(pred_logits.shape[0], -1),
            flat_t,
            pos_weight=pos_weight,
            reduction="mean",
        )

    def _generate_soft_targets(self, target_map: torch.Tensor):
        """
        Binarizes the claim-count map (any cell claimed by >=1 agent -> 1,
        i.e. per-cell "is this a target") and blurs it with a Gaussian filter
        on the GPU. The blur provides a smoother gradient signal than hard
        0/1 targets; sigma controls the hill width around each target.

        Blur of a {0,1} map with a non-negative kernel summing to 1 is a
        convex combination, so targets stay in [0,1] as BCE requires — raw
        counts would make the BCE loss unbounded below. Counts stay in the
        dataset; binarization happens only here, at train time.

        Boundary handling: zero padding truncates the kernel at the map edge,
        so a claim near a border would blur to less total mass than the same
        claim in the middle (position-dependent targets). We correct this by
        dividing each claimed cell by its in-bounds kernel mass (the conv of
        an all-ones map with the same kernel, evaluated at the claim) BEFORE
        the blur: each claim's truncated Gaussian hill is renormalized to
        exactly one unit of target mass, independent of position. The final
        clamp keeps the [0,1] guarantee for extreme corners at low sigma.
        target_map: (B, H, W)
        """
        target_map = (target_map > 0).float()
        kernel_size = int(2 * math.ceil(2 * self._sigma()) + 1)

        x = torch.arange(kernel_size, dtype=torch.float32, device=target_map.device)
        x = x - kernel_size // 2
        kernel_1d = torch.exp(-(x**2) / (2 * self._sigma() ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

        target_reshaped = target_map.unsqueeze(1)  # (B, 1, H, W)
        padding = kernel_size // 2

        # Divide each claim by its in-bounds kernel mass (conv of an all-ones
        # map with the same kernel) before blurring, so each claimed cell
        # contributes one unit of target mass regardless of how the kernel is
        # truncated at the map edge (see docstring).
        coverage = F.conv2d(
            torch.ones_like(target_reshaped), kernel_2d, padding=padding
        )
        weighted_claims = target_reshaped / coverage.clamp_min(1e-8)

        soft_targets = F.conv2d(weighted_claims, kernel_2d, padding=padding)
        # Extreme corners at low sigma can push the hill center past 1.
        return soft_targets.clamp(max=1.0).squeeze(1)  # (B, H, W)

    def pretrain_step(self, batch):
        x = batch["states"]
        history = batch["history"]
        target_map = batch["true_goal_map"]  # hindsight claim map (B, H, W)

        self.inference_model.train()
        pred_logits = self.forward(x, history, cached_features=False)  # (B, H, W)

        soft_targets = self._generate_soft_targets(target_map)
        loss = self._bce_loss(pred_logits, soft_targets)

        loss_val = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.inference_model.parameters(), 1.0)
        self.optimizer.step()

        # Soft target update
        with torch.no_grad():
            for param, target_param in zip(
                self.inference_model.parameters(), self.tgt_model.parameters()
            ):
                target_param.lerp_(param, self.args.tau_soft)

        g_map = torch.sigmoid(pred_logits)  # (B, H, W)
        mae = self.heatmap_mae(g_map, target_map)
        spatial_error = self.expected_spatial_error(g_map, target_map)

        return loss_val, mae, spatial_error

    def train_step(self, batch, cached_features=False):
        x = batch["states"]
        history = batch["history"]
        target_map = batch["true_goal_map"]  # hindsight claim map (B, H, W)
        self.inference_model.train()
        pred_logits = self.forward(x, history, cached_features)  # (B, H, W)

        soft_targets = self._generate_soft_targets(target_map)
        loss = self._bce_loss(pred_logits, soft_targets)

        loss_val = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.inference_model.parameters(), 1.0)
        self.optimizer.step()

        # Soft target update
        with torch.no_grad():
            for param, target_param in zip(
                self.inference_model.parameters(), self.tgt_model.parameters()
            ):
                target_param.lerp_(param, self.args.tau_soft)

        return loss_val


if __name__ == "__main__":
    wandb.init(mode="disabled", project="om-test")
    model = OpponentModel(SpatialOpponentModel(OMGArgs()), OMGArgs())
    dataset_path = "./dataset/dataset_map_3.pt"

    print("Loading dataset and pretraining OM...")
    dataset = torch.load(dataset_path, weights_only=False)
    model.pretrain(dataset, epochs=10, batch_size=128)
    del dataset
    wandb.finish()

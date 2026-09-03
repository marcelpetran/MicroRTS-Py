import math
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import wandb
from omexplore.models.transformers import SpatialOpponentModel
from omexplore.utils.omg_args import OMGArgs


class OpponentModel(nn.Module):
    def __init__(self, model: SpatialOpponentModel, args: OMGArgs = OMGArgs()):
        super(OpponentModel, self).__init__()
        self.inference_model = model.to(args.device)
        self.tgt_model = SpatialOpponentModel(args).to(args.device)
        self.tgt_model.load_state_dict(self.inference_model.state_dict())
        for param in self.tgt_model.parameters():
            param.requires_grad = False
        self.tgt_model.eval()
        # self.inference_model = torch.compile(self.inference_model)
        self.optimizer = torch.optim.Adam(
            self.inference_model.parameters(), lr=args.lr_om
        )
        self.device = args.device
        self.args = args

    def _sigma(self):
        """
        Computes the current sigma value based on the decay schedule.
        """
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
        # Add small value to prevent log(0)
        log_g_map = torch.log(g_map_flat + 1e-8)

        # Compute KL Divergence
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

    def expected_spatial_error(
        self, g_map: torch.Tensor, true_goal_map: torch.Tensor
    ) -> float:
        """
        Probability-weighted Manhattan distance from each cell to the nearest valid
        target, averaged over the batch. Vectorized over the batch dimension.

        Args:
            g_map: Predicted probabilities after softmax (B, H, W)
            true_goal_map: Ground truth probabilities (B, H, W)
        """
        B, H, W = g_map.shape
        dist = self._pairwise_manhattan(H, W, g_map.device)  # (HW, HW)

        g_flat = g_map.reshape(B, -1)  # (B, HW)
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
            epoch_kl_divs = []
            epoch_spatial_errors = []

            pbar = tqdm(
                range(0, len(dataset), batch_size), desc=f"Epoch {epoch + 1}/{epochs}"
            )

            for i in pbar:
                batch_data = dataset[i : i + batch_size]

                # Prepare batch data
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
                    "true_opp_heatmap": torch.from_numpy(
                        np.stack(
                            [b["true_opp_heatmap"] for b in batch_data],
                            dtype=np.float32,
                        )
                    ).to(self.device, non_blocking=True),
                }

                loss, kl_error, spatial_error = self.pretrain_step(om_batch)
                epoch_losses.append(loss)
                epoch_kl_divs.append(kl_error)
                epoch_spatial_errors.append(spatial_error)
                global_step += 1

                # Update progress bar suffix with current loss
                pbar.set_postfix({"loss": f"{loss:.4f}"})

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_kl_div = sum(epoch_kl_divs) / len(epoch_kl_divs)
            avg_spatial_error = sum(epoch_spatial_errors) / len(epoch_spatial_errors)
            print(f"  => Average Loss: {avg_loss:.6f}")

            # Log epoch-level metrics
            wandb.log(
                {
                    "pretrain/epoch_loss": avg_loss,
                    "pretrain/epoch_kl_divergence": avg_kl_div,
                    "pretrain/epoch_spatial_error": avg_spatial_error,
                    "epoch": epoch,
                }
            )

    def forward(
        self, x: torch.Tensor, history: Dict, cached_features=True
    ) -> torch.Tensor:
        """
        Calculates the forward pass, using the inference model
        to predict the opponent's subgoal.

        Args:
            x (Tensor): Current state s_t (B, H, W, F).
            history (Dict): Historical trajectory (states/opp_actions).

        Returns:
            Heatmap (B, H, W) of the predicted subgoal location.
        """
        self.inference_model.eval()  # Ensure the model is in evaluation mode
        return self.inference_model(x, history, cached_features=cached_features)

    def _generate_soft_targets(self, target_map: torch.Tensor):
        """
        Applies a Gaussian filter directly on the GPU using PyTorch Conv2d.
        This makes model learn faster and maybe even avoids getting stuck in local minima
        as it provides a smoother gradient signal compared to a hard one-hot target.
        The sigma parameter controls how much smoothing is applied,
        with higher values creating a wider "hill" around the true target location.
        target_map: (B, H, W)
        """
        kernel_size = int(2 * math.ceil(2 * self._sigma()) + 1)

        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size, dtype=torch.float32, device=target_map.device)
        x = x - kernel_size // 2
        kernel_1d = torch.exp(-(x**2) / (2 * self._sigma() ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Create 2D Gaussian kernel via outer product
        kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

        # Reshape target map for convolution: (B, C, H, W) where C=1
        target_reshaped = target_map.unsqueeze(1)

        # Apply padding to maintain spatial dimensions
        padding = kernel_size // 2
        soft_targets = F.conv2d(target_reshaped, kernel_2d, padding=padding)
        soft_targets = torch.clamp(soft_targets, min=0.0)
        # Normalize the soft targets to ensure they sum to 1 across the spatial dimensions
        soft_targets = soft_targets / soft_targets.sum(
            dim=(2, 3), keepdim=True
        ).clamp_min(1e-8)

        return soft_targets.squeeze(1)  # Return to (B, H, W)

    def pretrain_step(self, batch):
        x = batch["states"]
        history = batch["history"]
        # (B, H, W) Ground Truth from Hindsight
        target_map = batch["true_goal_map"]
        true_opp_heatmap = batch["true_opp_heatmap"]

        self.inference_model.train()
        pred_logits = self.forward(x, history, cached_features=False)  # (B, H, W)

        # Generate soft targets with Gaussian smoothing
        soft_targets = self._generate_soft_targets(target_map)
        soft_true_targets = self._generate_soft_targets(true_opp_heatmap)
        log_probs = F.log_softmax(pred_logits.view(pred_logits.shape[0], -1), dim=-1)
        target_dist = soft_targets.view(soft_targets.shape[0], -1)
        target_true_dist = soft_true_targets.view(soft_true_targets.shape[0], -1)
        loss = (
            F.kl_div(log_probs, target_true_dist, reduction="batchmean")
            + F.kl_div(log_probs, target_dist, reduction="batchmean") * 0.3
        )

        loss_val = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.inference_model.parameters(), 1.0)
        self.optimizer.step()

        with torch.no_grad():
            for param, target_param in zip(
                self.inference_model.parameters(), self.tgt_model.parameters()
            ):
                target_param.lerp_(param, self.args.tau_soft)

        opp_heatmap = batch["true_opp_heatmap"].to(self.device)
        g_map = F.softmax(pred_logits.view(pred_logits.shape[0], -1), dim=-1).view_as(
            pred_logits
        )  # (B, H, W)
        kl_div = self.heatmap_kl_divergence(g_map, opp_heatmap)
        spatial_error = self.expected_spatial_error(g_map, opp_heatmap)

        return loss_val, kl_div, spatial_error

    def train_step(self, batch, cached_features=False):
        x = batch["states"]
        history = batch["history"]
        # (B, H, W) Ground Truth from Hindsight
        target_map = batch["true_goal_map"]
        self.inference_model.train()
        pred_logits = self.forward(x, history, cached_features)  # (B, H, W)

        # Generate soft targets with Gaussian smoothing
        soft_targets = self._generate_soft_targets(target_map)
        log_probs = F.log_softmax(pred_logits.view(pred_logits.shape[0], -1), dim=-1)
        target_dist = soft_targets.view(soft_targets.shape[0], -1)
        loss = F.kl_div(log_probs, target_dist, reduction="batchmean")

        loss_val = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.inference_model.parameters(), 1.0)
        self.optimizer.step()

        # Target update (soft update)
        with torch.no_grad():
            for param, target_param in zip(
                self.inference_model.parameters(), self.tgt_model.parameters()
            ):
                target_param.lerp_(param, self.args.tau_soft)

        return loss_val


if __name__ == "__main__":
    wandb.init(mode="disabled", project="om-test")
    model = OpponentModel(SpatialOpponentModel(OMGArgs()), OMGArgs())
    dataset_path = f"./dataset/dataset_map_3.pt"

    print("Loading dataset and pretraining OM...")
    dataset = torch.load(dataset_path, weights_only=False)
    model.pretrain(dataset, epochs=10, batch_size=128)
    del dataset
    wandb.finish()

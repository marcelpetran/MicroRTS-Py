"""History-matched Classic baseline (Classic-H / DRQN-style control).

Purpose
-------
The unanimous reviewer objection (R1/R2/R3) is that the OM agent sees a
50-step fogged-observation history through a transformer while plain Classic
sees only the current observation, so the reported training stabilisation
could stem from history access rather than opponent-subgoal inference.

This agent is the control. It is the OM agent with the opponent-subgoal
*objective removed*:

  - Identical SpatialOpponentModel feature extractor + transformer encoder,
    reading the same rolling history (same buffers, masks, motion pairing).
  - The summary token is projected to an (H, W) latent map fused into the
    SAME QNet at the channel the OM heatmap occupies (same belief channels).
  - NO hindsight labels, NO subgoal KL loss: the encoder is shaped solely by
    the Q-learning gradient (end-to-end through Q).

The only remaining difference from the OM agent is opponent-subgoal
supervision on the shared encoder — exactly what the reviewers want isolated.
If Classic-H stabilises where Classic collapses, stability is largely a
temporal-representation effect; if Classic-H still collapses, stabilisation is
attributable to the subgoal objective.

Two encoder modes, identical network:
  - acting/eval: cached_features=True (reuses the rollout's feature buffers).
  - training:    cached_features=False (re-embeds raw history so the feature
                 extractor receives gradients).

Inherits the OM agent's rollout, n-step returns and history bookkeeping.
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from omexplore.agents.q_agent import QLearningAgent
from omexplore.models.opponent_model import OpponentModel
from omexplore.utils.omg_args import OMGArgs


class HistoryEncoder(nn.Module):
    """Q-driven history encoder sharing the OM encoder's exact architecture.

    Repurposes SpatialOpponentModel.spatial_head (d_model -> H*W) as a learned
    projection of the summary token. Here it is NOT an opponent-subgoal heatmap;
    it is an unsupervised spatial latent fused into Q. Only the Q-loss shapes it.
    """

    def __init__(self, model: OpponentModel, args: OMGArgs):
        super().__init__()
        self.om = model.inference_model
        self.args = args

    def forward(self, x, history, cached_features: bool) -> torch.Tensor:
        om = self.om
        B, H, W, _ = x.shape
        hist_mask = history["mask"]
        T = hist_mask.shape[1]

        if cached_features:
            hist_feat = history["state_features"]
            x_prev = history.get("prev_obs", torch.zeros_like(x))
        else:
            hist_states = history["states"]
            prev_states = torch.zeros_like(hist_states)
            prev_states[:, 1:] = hist_states[:, :-1]
            prev_first = history.get("prev_first")
            if prev_first is None:
                prev_first = torch.zeros_like(hist_states[:, 0])
            prev_states[:, 0] = prev_first
            hist_flat = hist_states.reshape(B * T, H, W, -1)
            prev_flat = prev_states.reshape(B * T, H, W, -1)
            hist_feat = om.get_features(hist_flat, prev_flat).reshape(B, T, -1)
            x_prev = hist_states[:, -1] * hist_mask[:, -1].view(B, 1, 1, 1).float()

        x_feat = om.get_features(x, x_prev).unsqueeze(1)
        seq = torch.cat([x_feat, hist_feat], dim=1) * np.sqrt(self.args.d_model)
        seq = om.pos_encoder(seq)
        x_mask = torch.ones((B, 1), dtype=torch.bool, device=x.device)
        full_mask = torch.cat([x_mask, hist_mask], dim=1)
        memory = om.transformer(seq, src_key_padding_mask=~full_mask)
        summary = memory[:, 0, :]  # (B, d_model)
        logits = om.spatial_head(summary)  # (B, H*W)
        return logits.view(B, H, W)


class QLearningAgentClassicH(QLearningAgent):
    """Classic DQN with a history transformer, no opponent-subgoal supervision."""

    def __init__(self, env, opponent_model: OpponentModel, args: OMGArgs):
        super().__init__(env, opponent_model, args)
        self.encoder = HistoryEncoder(opponent_model, args).to(self.device)
        # One optimizer over Q-net + encoder: history shaped ONLY by Q-objective.
        self.opt = torch.optim.Adam(
            list(self.q.parameters()) + list(self.encoder.parameters()),
            lr=self.args.lr,
            eps=1e-6,
        )

    @torch.no_grad()
    def select_action(self, s_t, s_aug, history, eval=False):
        x = torch.from_numpy(s_t).float().unsqueeze(0).to(self.device)
        x_aug = torch.from_numpy(s_aug).float().unsqueeze(0).to(self.device)
        # Acting must be deterministic: the encoder (pos_encoder + transformer)
        # applies dropout in train mode, which would noisify every action. Force
        # no-dropout for acting only; update() puts the encoder back in train().
        self.encoder.om.eval()
        g_latent = self.encoder(x, history, cached_features=True)  # (1, H, W)
        qvals = self.q(x_aug, g_latent)
        tau = 0.05 if eval else self._tau()
        entropy = Categorical(logits=qvals / 0.05).entropy().item()
        a = self.choose_action(qvals, tau, eval)
        # Key name kept for the shared rollout signature; this is a learned
        # latent map, not a supervised subgoal heatmap.
        return a, g_latent.squeeze(0), entropy

    def compute_targets(self, batch: List[Dict], history):
        s = torch.from_numpy(
            np.array([b["state"] for b in batch], dtype=np.float32)
        ).to(self.device)
        squ = (
            torch.from_numpy(np.stack([b["state_aug"] for b in batch]))
            .float()
            .to(self.device)
        )
        sp = torch.from_numpy(
            np.array([b["next_state_n"] for b in batch], dtype=np.float32)
        ).to(self.device)
        spu = (
            torch.from_numpy(np.stack([b["next_state_aug_n"] for b in batch]))
            .float()
            .to(self.device)
        )
        a = torch.from_numpy(np.array([b["action"] for b in batch], dtype=np.int64)).to(
            self.device
        )
        r = torch.from_numpy(
            np.array([b["n_reward"] for b in batch], dtype=np.float32)
        ).to(self.device)
        done = torch.from_numpy(
            np.array([b["done_n"] for b in batch], dtype=np.float32)
        ).to(self.device)

        # Raw-history recompute so gradients reach the feature extractor.
        cur_hist = self.model.collate_history(batch)
        g_cur = self.encoder(s, cur_hist, cached_features=False)  # (B, H, W), grad
        hist_next = self.model.collate_history(batch, len_key="hist_len_n")
        with torch.no_grad():
            g_next = self.encoder(sp, hist_next, cached_features=False)
        q_sa = self.q(squ, g_cur).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_val = self.q(spu, g_next)
            best = (q_val + torch.rand_like(q_val) * 1e-6).argmax(dim=1, keepdim=True)
            q_next = self.q_tgt(spu, g_next).gather(1, best).squeeze(1)
            target = r + (1.0 - done) * (self.args.gamma**self.args.n_step) * q_next
            target = torch.clamp(target, min=-15.0, max=15.0)
        return q_sa, target

    def update(self):
        if len(self.replay) < self.args.min_replay:
            return (None, None)
        if self.global_step % self.args.train_every != 0:
            return (None, None)
        # Encoder trains ONLY here (dropout on for representation regularisation).
        self.encoder.om.train()
        batch_list = self.replay.sample(self.args.batch_size)
        hist = self.model.collate_history(batch_list)
        q_sa, target = self.compute_targets(batch_list, hist)
        loss = F.smooth_l1_loss(q_sa, target, reduction="mean")
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        params = list(self.q.parameters()) + list(self.encoder.parameters())
        nn.utils.clip_grad_norm_(params, 5.0)
        self.opt.step()
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.q_tgt.parameters()):
                tp.lerp_(p, self.args.tau_soft)
        # No subgoal objective; keep the OM-loss log slot readable.
        return loss.item(), 0.0

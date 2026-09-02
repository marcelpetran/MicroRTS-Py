import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import wandb
from omexplore.models.beliefs import BeliefTracker
from omexplore.models.buffers import ReplayBuffer, ReservoirBuffer
from omexplore.models.networks import QNet, SLnet
from omexplore.utils.omg_args import OMGArgs


class FSPAgentOM:
    """
    Unified Fictitious Self-Play Agent with Opponent Modeling.
    Contains both RL (Best Response) and SL (Average Strategy) components.

    Refactored to use the POSG env API:
      - Belief-augmented states for Q-learning (QNet expects belief channels).
      - State-only history (no opponent actions) for the OM transformer, which
        uses motion encoding via get_features(x, x_prev).
      - Memory-efficient transition storage: a shared episode states array is
        referenced by every transition instead of per-step feature tensors.
    """

    def __init__(self, env, opponent_model, args: OMGArgs = OMGArgs()):
        self.env = env
        self.model = opponent_model
        self.args = args
        self.device = torch.device(args.device)

        if args.state_shape is None:
            obs = self.env.reset()
            H, W, F_dim = obs[0].shape
            self.args.state_shape = (H, W, F_dim)

        if not hasattr(self.env, "action_space"):
            raise ValueError("Env must have action_space (list or int).")
        self.args.action_dim = (
            len(self.env.action_space)
            if hasattr(self.env.action_space, "__len__")
            else self.env.action_space.n
        )

        # RL Networks & Optimizer
        self.q = QNet(args).to(self.device)
        self.q_tgt = QNet(args).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.opt_rl = torch.optim.Adam(self.q.parameters(), lr=self.args.lr)

        # SL Network & Optimizer
        self.sl = SLnet(args).to(self.device)
        self.opt_sl = torch.optim.Adam(self.sl.parameters(), lr=self.args.lr)

        # FSP Buffers
        self.rl_replay = ReplayBuffer(self.args.capacity)
        self.sl_replay = ReservoirBuffer(self.args.sl_capacity)

        # Belief tracker (player-0 perspective; per-perspective trackers are
        # created inside run_fsp_episode / run_test_episode for self-play)
        self.tracker = BeliefTracker(
            self.args.H,
            self.args.W,
            map_layout=self.env.map_layout,
            horizon=self.args.max_steps,
        )

        self.global_step = 0
        self.is_frozen_as_sl = False

    def reset(self):
        pass

    def freeze_as_sl_opponent(self):
        """Forces the agent to only act using the Average Strategy."""
        self.is_frozen_as_sl = True

    # ------------- Tau Schedules --------------

    def _tau(self) -> float:
        t = min(self.global_step, self.args.tau_decay_steps)
        return self.args.tau_end + (self.args.tau_start - self.args.tau_end) * (
            1 - t / self.args.tau_decay_steps
        )

    # ------------- Action Selection Methods -------------

    def _choose_q_action(self, qvals: torch.Tensor, beta: float, eval=False) -> int:
        gumbel_noise = -beta * torch.empty_like(qvals).exponential_().log()
        if eval:
            dist = F.softmax(qvals / beta, dim=-1)
            return int(torch.multinomial(dist, num_samples=1).item())
        return int(torch.argmax(qvals + gumbel_noise))

    @torch.no_grad()
    def select_rl_action(
        self,
        s_t: np.ndarray,
        s_aug: np.ndarray,
        history: Dict[str, torch.Tensor],
        eval=False,
    ) -> tuple[int, torch.Tensor, float]:
        """Best Response action using Q-learning and OM.

        Args:
          s_t: raw observation (H, W, F) — fed to the OM transformer.
          s_aug: belief-augmented observation — fed to QNet.
          history: OM history dict with state_features / mask / prev_obs.
        """
        self.q.eval()
        x = torch.from_numpy(s_t).float().unsqueeze(0).to(self.device)
        x_aug = torch.from_numpy(s_aug).float().unsqueeze(0).to(self.device)

        g_logits = self.model(x, history)  # OM uses raw state (cached_features=True)
        g_map = F.softmax(g_logits.view(g_logits.shape[0], -1), dim=-1).view_as(
            g_logits
        )

        qvals = self.q(x_aug, g_map)  # QNet uses belief-augmented state
        tau = 0.05 if eval else self._tau()
        entropy = Categorical(logits=qvals / tau).entropy().item()

        a = self._choose_q_action(qvals, tau, eval)
        self.q.train()
        return a, g_map.squeeze(0), entropy

    @torch.no_grad()
    def select_sl_action(self, s_t: np.ndarray, eval=False) -> tuple[int, float]:
        """Average Strategy action using Supervised Learning (raw state)."""
        self.sl.eval()
        s = torch.from_numpy(s_t).float().unsqueeze(0).to(self.device)
        logits = self.sl(s)
        entropy = 0.0

        if eval:
            action = torch.argmax(logits, dim=1).item()
        else:
            dist = Categorical(logits=logits)
            action = dist.sample().item()
            entropy = dist.entropy().item()

        self.sl.train()
        return action, entropy

    def select_action(
        self,
        s_t: np.ndarray,
        s_aug: np.ndarray = None,
        history: Dict[str, torch.Tensor] = None,
        eval=False,
    ) -> tuple[int, float]:
        """Generic interface for opponents. Defaults to SL if frozen."""
        if self.is_frozen_as_sl:
            return self.select_sl_action(s_t, eval=eval)
        a, _, ent = self.select_rl_action(s_t, s_aug, history, eval=eval)
        return a, ent

    # ------------- RL Update Logic -------------

    def compute_targets(
        self, batch: List[Dict], history: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DDQN target computation using HER goal maps (mirrors QLearningAgent)."""
        s = torch.from_numpy(
            np.array([b["state"] for b in batch], dtype=np.float32)
        ).to(self.device)
        squ = (
            torch.from_numpy(np.stack([b["state_aug"] for b in batch]))
            .float()
            .to(self.device)
        )
        sp = torch.from_numpy(
            np.array([b["next_state"] for b in batch], dtype=np.float32)
        ).to(self.device)
        spu = (
            torch.from_numpy(np.stack([b["next_state_aug"] for b in batch]))
            .float()
            .to(self.device)
        )
        a = torch.from_numpy(np.array([b["action"] for b in batch], dtype=np.int64)).to(
            self.device
        )
        r = torch.from_numpy(
            np.array([b["reward"] for b in batch], dtype=np.float32)
        ).to(self.device)
        done = torch.from_numpy(
            np.array([b["done"] for b in batch], dtype=np.float32)
        ).to(self.device)

        with torch.no_grad():
            hist = history
            g_logits = self.model.tgt_model(s, hist, cached_features=False)
            g_map = F.softmax(g_logits.view(len(batch), -1), dim=-1).view_as(g_logits)

            # Build next-state history by shifting and appending current state
            hist_states = history["states"].clone()  # (B, max_len, H, W, F)
            hist_mask = history["mask"].clone()  # (B, max_len)

            hist_states[:, :-1] = hist_states[:, 1:]
            hist_mask[:, :-1] = hist_mask[:, 1:]
            hist_states[:, -1] = s
            hist_mask[:, -1] = True

            hist_next = {"states": hist_states, "mask": hist_mask}
            g_logits_next = self.model.tgt_model(sp, hist_next, cached_features=False)
            g_map_next = F.softmax(g_logits_next.view(len(batch), -1), dim=-1).view_as(
                g_logits_next
            )

        # Helper log: KL between live and EMA (target) OM predictions
        with torch.no_grad():
            self.model.inference_model.eval()
            g_logits_live = self.model.inference_model(s, hist, cached_features=False)
            self.model.inference_model.train()
            g_live = F.softmax(g_logits_live.view(len(batch), -1), dim=-1)
            g_ema = F.softmax(g_logits.view(len(batch), -1), dim=-1)
            kl_live_ema = (
                (g_live * (torch.log(g_live + 1e-8) - torch.log(g_ema + 1e-8)))
                .sum(dim=-1)
                .mean()
                .item()
            )
        wandb.log({"om/kl_live_ema": kl_live_ema}, step=self.global_step)

        # 1. Q(s, g, a)
        q_sa = self.q(squ, g_map).gather(1, a.unsqueeze(1)).squeeze(1)

        # 2. Target = r + gamma * max_a' Q_tgt(s', g_next, a')
        with torch.no_grad():
            q_val = self.q(spu, g_map_next)
            noise = torch.rand_like(q_val) * 1e-6
            best_actions = (q_val + noise).argmax(dim=1, keepdim=True)
            q_next = self.q_tgt(spu, g_map_next).gather(1, best_actions).squeeze(1)
            target = r + (1.0 - done) * self.args.gamma * q_next
            target = torch.clamp(target, min=-15.0, max=15.0)

        return q_sa, target

    def update_rl(self):
        """Updates the Q-network and the OM Transformer."""
        if len(self.rl_replay) < self.args.min_replay:
            return None, None

        batch_list = self.rl_replay.sample(self.args.batch_size)

        # Build the OM batch with collated state histories (no opponent actions)
        om_batch = {
            "states": torch.from_numpy(
                np.array([b["state"] for b in batch_list], dtype=np.float32)
            ).to(self.device),
            "history": self.model.collate_history(batch_list),
            "true_goal_map": torch.from_numpy(
                np.array([b["true_goal_map"] for b in batch_list], dtype=np.float32)
            ).to(self.device),
        }

        # Update Q-Network
        q_sa, target = self.compute_targets(batch_list, om_batch["history"])
        loss = F.smooth_l1_loss(q_sa, target, reduction="mean")
        loss_val = loss.item()

        self.opt_rl.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt_rl.step()

        # Target network soft update
        with torch.no_grad():
            for param, target_param in zip(
                self.q.parameters(), self.q_tgt.parameters()
            ):
                target_param.lerp_(param, self.args.tau_soft)

        # Update OM Transformer (cached_features=False: recompute from raw states)
        model_loss = self.model.train_step(om_batch, cached_features=False)

        return loss_val, model_loss

    # ------------- SL Update Logic -------------

    def update_sl(self):
        """Updates the SL Average Strategy network."""
        if len(self.sl_replay) < self.args.min_replay:
            return None

        batch = self.sl_replay.sample(self.args.batch_size)
        s = (
            torch.from_numpy(np.stack([b["state"] for b in batch]))
            .float()
            .to(self.device)
        )
        a = torch.from_numpy(np.array([b["action"] for b in batch], dtype=np.int64)).to(
            self.device
        )

        logits = self.sl(s)
        loss = F.cross_entropy(logits, a)

        self.opt_sl.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.sl.parameters(), 5.0)
        self.opt_sl.step()

        return loss.item()

    # ------------- Hindsight Relabeling -------------

    def _apply_hindsight_relabeling(self, episode_transitions: List, H: int, W: int):
        """HER labeling: assign true_goal_map based on opponent's achieved subgoals."""
        current_true_goal_pos = None

        # 1. Hindsight labeling for truncated episodes
        if len(episode_transitions) > 0:
            final_t = episode_transitions[-1]
            if final_t["opp_reward"] == 0:
                opp_pos_arr = np.argwhere(final_t["global_state"][:, :, 3] == 1)
                if len(opp_pos_arr) > 0:
                    current_true_goal_pos = tuple(opp_pos_arr[0])

        # 2. Walk backward through the episode to label goals
        for t in reversed(episode_transitions):
            if t["opp_reward"] > 0:
                opp_pos_indices = np.argwhere(t["next_global_state"][:, :, 3] == 1)
                if len(opp_pos_indices) > 0:
                    current_true_goal_pos = tuple(opp_pos_indices[0])

            true_map = np.zeros((H, W), dtype=np.float32)
            if current_true_goal_pos is not None:
                true_map[current_true_goal_pos[0], current_true_goal_pos[1]] = 1.0

            t["true_goal_map"] = true_map
            del t["opp_reward"]

    # ------------- Data Generation (Rollout) -------------

    def run_fsp_episode(
        self, opponent_agent, eta: float, max_steps: int = 500
    ) -> Dict[str, float]:
        """
        Self-play rollout mixing RL and SL policies using eta. Both players
        share this agent's networks; each maintains its own belief tracker and
        OM history (state-only, no opponent actions).

        Stores memory-efficient transitions: a shared episode states array is
        referenced by every transition (cf. per-step feature tensors before).
        """
        obs = self.env.reset()
        if random.random() < 0.3:
            obs = self.env.reset_random_spawn()
        elif random.random() < 0.5:
            obs = self.env.swap_agents()
        opponent_agent.reset()

        # Per-perspective belief trackers
        tracker_0 = BeliefTracker(
            self.args.H,
            self.args.W,
            map_layout=self.env.map_layout,
            horizon=self.args.max_steps,
        )
        tracker_1 = BeliefTracker(
            self.args.H,
            self.args.W,
            map_layout=self.env.map_layout,
            horizon=self.args.max_steps,
        )
        tracker_0.reset(use_map_prior=self.args.belief_map_prior)
        tracker_1.reset(use_map_prior=self.args.belief_map_prior)
        tracker_0.update(obs[0])
        tracker_1.update(obs[1])

        done = False
        ep_ret, opp_ret, rl_ep_entropy, sl_ep_entropy = 0.0, 0.0, 0.0, 0.0

        history_len = self.args.max_history_length
        # Player-0 OM history (state features + mask + prev_obs for motion encoding)
        rolling_feats = torch.zeros(
            (1, history_len, self.args.d_model), device=self.device
        )
        rolling_mask = torch.zeros(
            (1, history_len), dtype=torch.bool, device=self.device
        )
        prev_state_tensor = torch.zeros((1, *obs[0].shape), device=self.device)
        current_seq_len = 0
        # Player-1 OM history
        opp_rolling_feats = torch.zeros(
            (1, history_len, self.args.d_model), device=self.device
        )
        opp_rolling_mask = torch.zeros(
            (1, history_len), dtype=torch.bool, device=self.device
        )
        opp_prev_state_tensor = torch.zeros((1, *obs[1].shape), device=self.device)
        opp_current_seq_len = 0

        episode_transitions = []
        ep_states = []
        H, W, _ = obs[0].shape

        for step in range(max_steps):
            history_gpu = {
                "state_features": rolling_feats,
                "mask": rolling_mask,
                "prev_obs": prev_state_tensor,
            }
            opp_history_gpu = {
                "state_features": opp_rolling_feats,
                "mask": opp_rolling_mask,
                "prev_obs": opp_prev_state_tensor,
            }

            # Player 0: compute RL (Best Response) and SL (Average) actions
            s_aug = tracker_0.augment(obs[0])
            rl_a, g_map, rl_entropy = self.select_rl_action(obs[0], s_aug, history_gpu)
            sl_a, sl_entropy = self.select_sl_action(obs[0])
            rl_ep_entropy += rl_entropy
            sl_ep_entropy += sl_entropy

            # Mix policies based on eta (Fictitious Play)
            if random.random() < eta:
                a = rl_a
                is_rl = True
            else:
                a = sl_a
                is_rl = False

            # Player 1 (opponent) acts
            opp_s_aug = tracker_1.augment(obs[1])
            if (
                hasattr(opponent_agent, "is_frozen_as_sl")
                and opponent_agent.is_frozen_as_sl
            ):
                a_opponent, _ = opponent_agent.select_sl_action(obs[1])
                opp_is_rl = False
            else:
                opp_rl_a, _, _ = opponent_agent.select_rl_action(
                    obs[1], opp_s_aug, opp_history_gpu
                )
                opp_sl_a, _ = opponent_agent.select_sl_action(obs[1])
                if random.random() < eta:
                    a_opponent = opp_rl_a
                    opp_is_rl = True
                else:
                    a_opponent = opp_sl_a
                    opp_is_rl = False

            actions = {0: a, 1: a_opponent}

            global_state = self.env.get_global_state()
            next_obs, reward, done, info = self.env.step(actions)
            next_global_state = self.env.get_global_state()

            tracker_0.update(next_obs[0])
            tracker_1.update(next_obs[1])
            next_aug = tracker_0.augment(next_obs[0])

            # SL buffer: only Best-Response actions (NFSP standard), raw state
            if is_rl:
                self.sl_replay.push({"state": obs[0].copy(), "action": a})
            if opp_is_rl and opponent_agent is self:
                self.sl_replay.push({"state": obs[1].copy(), "action": a_opponent})

            transition = {
                "state": obs[0].copy(),
                "state_aug": s_aug.copy(),
                "global_state": global_state.copy(),
                "action": a,
                "reward": float(reward[0]),
                "opp_reward": float(reward[1]),
                "next_state": next_obs[0].copy(),
                "next_state_aug": next_aug.copy(),
                "next_global_state": next_global_state.copy(),
                "done": bool(done),
                "hist_len": len(ep_states),
            }
            ep_states.append(obs[0].copy())
            episode_transitions.append(transition)

            # Update player-0 OM history (state-only, motion-encoded features)
            state_tensor = torch.from_numpy(obs[0]).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                new_feat = self.model.inference_model.get_features(
                    state_tensor, prev_state_tensor
                )
            rolling_feats = torch.roll(rolling_feats, shifts=-1, dims=1)
            rolling_mask = torch.roll(rolling_mask, shifts=-1, dims=1)
            rolling_feats[:, -1, :] = new_feat
            if current_seq_len < history_len:
                current_seq_len += 1
            rolling_mask[:, -current_seq_len:] = True
            prev_state_tensor = state_tensor

            # Update player-1 OM history
            opp_state_tensor = (
                torch.from_numpy(obs[1]).float().unsqueeze(0).to(self.device)
            )
            with torch.no_grad():
                opp_new_feat = self.model.inference_model.get_features(
                    opp_state_tensor, opp_prev_state_tensor
                )
            opp_rolling_feats = torch.roll(opp_rolling_feats, shifts=-1, dims=1)
            opp_rolling_mask = torch.roll(opp_rolling_mask, shifts=-1, dims=1)
            opp_rolling_feats[:, -1, :] = opp_new_feat
            if opp_current_seq_len < history_len:
                opp_current_seq_len += 1
            opp_rolling_mask[:, -opp_current_seq_len:] = True
            opp_prev_state_tensor = opp_state_tensor

            ep_ret += reward[0]
            opp_ret += reward[1]
            obs = next_obs
            self.global_step += 1

            if done:
                break

        self._apply_hindsight_relabeling(episode_transitions, H, W)

        # Share the episode states array across all transitions (memory-efficient)
        states_arr = np.stack(ep_states)
        for t in episode_transitions:
            t["history"] = {"states": states_arr}
            self.rl_replay.push(t)

        return {
            "return": ep_ret,
            "steps": (step + 1) if max_steps > 0 else 0,
            "opp_return": opp_ret,
            "avg_rl_entropy": rl_ep_entropy / (step + 1) if max_steps > 0 else 0.0,
            "avg_sl_entropy": sl_ep_entropy / (step + 1) if max_steps > 0 else 0.0,
        }

    def run_test_episode(
        self, opponent_agent, use_sl: bool = True, max_steps: int = 500
    ) -> Dict[str, float]:
        """Evaluation rollout. Defaults to Average Strategy (SL) as per FSP standards."""
        self.model.inference_model.eval()
        obs = self.env.reset()
        opponent_agent.reset()

        tracker_0 = BeliefTracker(
            self.args.H,
            self.args.W,
            map_layout=self.env.map_layout,
            horizon=self.args.max_steps,
        )
        tracker_1 = BeliefTracker(
            self.args.H,
            self.args.W,
            map_layout=self.env.map_layout,
            horizon=self.args.max_steps,
        )
        tracker_0.reset(use_map_prior=self.args.belief_map_prior)
        tracker_1.reset(use_map_prior=self.args.belief_map_prior)
        tracker_0.update(obs[0])
        tracker_1.update(obs[1])

        done = False
        ep_ret, opp_ret, rl_ep_entropy, sl_ep_entropy = 0.0, 0.0, 0.0, 0.0

        history_len = self.args.max_history_length
        rolling_feats = torch.zeros(
            (1, history_len, self.args.d_model), device=self.device
        )
        rolling_mask = torch.zeros(
            (1, history_len), dtype=torch.bool, device=self.device
        )
        prev_state_tensor = torch.zeros((1, *obs[0].shape), device=self.device)
        current_seq_len = 0
        opp_rolling_feats = torch.zeros(
            (1, history_len, self.args.d_model), device=self.device
        )
        opp_rolling_mask = torch.zeros(
            (1, history_len), dtype=torch.bool, device=self.device
        )
        opp_prev_state_tensor = torch.zeros((1, *obs[1].shape), device=self.device)
        opp_current_seq_len = 0

        for step in range(max_steps):
            history_gpu = {
                "state_features": rolling_feats,
                "mask": rolling_mask,
                "prev_obs": prev_state_tensor,
            }
            opp_history_gpu = {
                "state_features": opp_rolling_feats,
                "mask": opp_rolling_mask,
                "prev_obs": opp_prev_state_tensor,
            }

            s_aug = tracker_0.augment(obs[0])
            sl_a, sl_entropy = self.select_sl_action(obs[0], eval=True)
            rl_a, _, rl_entropy = self.select_rl_action(
                obs[0], s_aug, history_gpu, eval=True
            )
            a = sl_a if use_sl else rl_a

            opp_s_aug = tracker_1.augment(obs[1])
            if (
                hasattr(opponent_agent, "is_frozen_as_sl")
                and opponent_agent.is_frozen_as_sl
            ):
                a_opponent, _ = opponent_agent.select_sl_action(obs[1], eval=True)
            elif isinstance(opponent_agent, FSPAgentOM):
                a_opponent, _ = opponent_agent.select_action(
                    obs[1], opp_s_aug, opp_history_gpu, eval=True
                )
            else:
                a_opponent, _ = opponent_agent.select_action(obs[1], eval=True)

            actions = {0: a, 1: a_opponent}
            next_obs, reward, done, info = self.env.step(actions)

            tracker_0.update(next_obs[0])
            tracker_1.update(next_obs[1])

            state_tensor = torch.from_numpy(obs[0]).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                new_feat = self.model.inference_model.get_features(
                    state_tensor, prev_state_tensor
                )
            rolling_feats = torch.roll(rolling_feats, shifts=-1, dims=1)
            rolling_mask = torch.roll(rolling_mask, shifts=-1, dims=1)
            rolling_feats[:, -1, :] = new_feat
            if current_seq_len < history_len:
                current_seq_len += 1
            rolling_mask[:, -current_seq_len:] = True
            prev_state_tensor = state_tensor

            opp_state_tensor = (
                torch.from_numpy(obs[1]).float().unsqueeze(0).to(self.device)
            )
            with torch.no_grad():
                opp_new_feat = self.model.inference_model.get_features(
                    opp_state_tensor, opp_prev_state_tensor
                )
            opp_rolling_feats = torch.roll(opp_rolling_feats, shifts=-1, dims=1)
            opp_rolling_mask = torch.roll(opp_rolling_mask, shifts=-1, dims=1)
            opp_rolling_feats[:, -1, :] = opp_new_feat
            if opp_current_seq_len < history_len:
                opp_current_seq_len += 1
            opp_rolling_mask[:, -opp_current_seq_len:] = True
            opp_prev_state_tensor = opp_state_tensor

            ep_ret += reward[0]
            opp_ret += reward[1]
            obs = next_obs
            rl_ep_entropy += rl_entropy
            sl_ep_entropy += sl_entropy

            if done:
                break

        return {
            "return": ep_ret,
            "steps": (step + 1) if max_steps > 0 else 0,
            "opp_return": opp_ret,
            "avg_rl_entropy": rl_ep_entropy / (step + 1)
            if max_steps > 0 and not use_sl
            else 0.0,
            "avg_sl_entropy": sl_ep_entropy / (step + 1)
            if max_steps > 0 and use_sl
            else 0.0,
        }

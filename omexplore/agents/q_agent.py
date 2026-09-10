import random
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.types import Number

from omexplore.envs.roadmap_foraging_env import TeamRoadmapEnv
from omexplore.models.beliefs import BeliefTracker
from omexplore.models.buffers import ReplayBuffer
from omexplore.models.networks import QNet
from omexplore.models.opponent_model import OpponentModel
from omexplore.utils.labeling import claim_count_map, compute_agent_goals
from omexplore.utils.omg_args import OMGArgs


class QLearningAgent:
    """
    Q-learning agent with Hindsight Experience Replay and subgoal inference
    for opponent/teammate modeling, for TeamRoadmapEnv.

    One SHARED Q-net controls every member of the learning team (team 0);
    the hostile teams are controlled externally (scripted TeamAgent or a
    future self-play agent). Two OMs are trained alongside:
      - self.model:     hostile OM, predicts the pooled hostile-team claim map
      - self.team_model: friendly OM, predicts the teammates' claim map
        (every team member except the acting agent; the agent decides its
        own goal, it only needs to predict the others)
    Labels are per-team claim COUNT maps from hindsight (see
    _apply_hindsight_relabeling), so a cell can hold > 1 when several agents
    aim at the same goal.

    Transitions store the raw int8 obs plus the team-level belief channels
    separately ("belief"/"next_belief"); the augmented CNN input is
    reconstructed at train time to keep the replay buffer small on the
    large maps.
    """

    def __init__(
        self,
        env: TeamRoadmapEnv,
        opponent_model: OpponentModel,
        team_model: OpponentModel,
        args: OMGArgs = OMGArgs(),
    ):
        self.env: TeamRoadmapEnv = env
        self.model: OpponentModel = opponent_model
        self.team_model: OpponentModel = team_model
        self.args: OMGArgs = args
        self.device: torch.device = torch.device(args.device)

        if not hasattr(self.env, "action_space"):
            raise ValueError("Env must have action_space (list or int).")
        self.args.action_dim = (
            len(self.env.action_space)
            if hasattr(self.env.action_space, "__len__")
            else self.env.action_space.n
        )

        # Team bookkeeping: the shared Q-net controls the whole learning
        # team (team 0); every other team is pooled into one hostile OM.
        self.learn_ids = self.env.get_team_members(0)
        self.hostile_ids = [a for a in self.env.agents if self.env.teams[a] != 0]

        # Networks
        self.q = QNet(args).to(self.device)
        self.q_tgt = QNet(args).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=self.args.lr, eps=1e-6)

        # Replay
        self.replay = ReplayBuffer(self.args.capacity)

        # Belief map (team-level: the team obs already pools team vision).
        # Channel indices depend on the obs layout: (1, 4, 6) for the
        # 7-channel team obs, (1, 3, 5) for the old 6-channel 1v1 obs.
        features = getattr(self.env, "features", 6)
        self.tracker = BeliefTracker(
            self.env.height,
            self.env.width,
            map_layout=self.env.map_layout,
            horizon=self.args.max_steps,
            channels=(1, 4, 6) if features == 7 else (1, 3, 5),
        )

        # Schedules
        self.global_step = 0

    def reset(self):
        pass

    # ------------- tau schedule -------------

    def _tau(self) -> float:
        t = min(self.global_step, self.args.tau_decay_steps)
        return self.args.tau_end + (self.args.tau_start - self.args.tau_end) * (
            1 - t / self.args.tau_decay_steps
        )

    # ------------- evaluation -------------

    @torch.no_grad()
    def value(self, s_t: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Q(s_t, g) -> (1, A); s_t: (1, H, W, F), g: (1, latent_dim)."""
        self.q.eval()
        return self.q(s_t, g)

    # ------------- visualization -------------
    @torch.no_grad()
    def heatmap_q_values(
        self, g: torch.Tensor, filename: str = "q_heatmap.png", save: bool = True
    ):
        """Visualize max-Q and the greedy action per grid cell.

        Teleports the anchor agent to every free cell and evaluates Q with the
        fixed subgoal g (an approximation: g is only valid at the agent's real
        position, but per-cell subgoals would be too expensive).
        g: (latent_dim) or (1, latent_dim).
        """
        self.q.eval()
        H, W, _ = self.args.state_shape
        g = g.unsqueeze(0)  # (1, latent_dim)

        q_value_map = np.zeros((H, W))
        policy_map = np.zeros((H, W))

        anchor = self.learn_ids[0]
        original_pos = self.env.agents[anchor]
        for pos in self.env._get_freed_positions() + [original_pos]:
            r, c = pos

            self.env.agents[anchor] = pos
            temp_state = self.env._get_observations()[anchor]

            s_tensor = (
                torch.from_numpy(
                    np.concatenate(
                        [
                            temp_state,
                            np.zeros((H, W, self.args.belief_channels), np.float32),
                        ],
                        -1,
                    )
                )
                .float()
                .unsqueeze(0)
                .to(self.device)
            )

            q_values = self.q(s_tensor, g)  # (1, num_actions)

            max_q_val, best_action = torch.max(q_values, dim=1)
            q_value_map[r, c] = max_q_val.item()
            policy_map[r, c] = best_action.item()

        # Restore the agent's original position
        self.env.agents[anchor] = original_pos
        opp_pos = self.env.agents[self.hostile_ids[0]]
        food_pos = self.env.food_positions
        wall_pos = self.env.walls

        # --- Plotting ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.scatter(
            original_pos[1],
            original_pos[0],
            color="blue",
            marker="X",
            s=100,
            label="Agent",
        )
        ax1.scatter(
            opp_pos[1], opp_pos[0], color="red", marker="X", s=100, label="Opponent"
        )
        if food_pos:
            food_x = [pos[1] for pos in food_pos]
            food_y = [pos[0] for pos in food_pos]
            ax1.scatter(food_x, food_y, color="green", marker="o", s=50, label="Food")
        if wall_pos:
            wall_x = [pos[1] for pos in wall_pos]
            wall_y = [pos[0] for pos in wall_pos]
            ax1.scatter(wall_x, wall_y, color="black", marker="s", s=50, label="Wall")
        # Q-value heatmap
        im1 = ax1.imshow(q_value_map, cmap="viridis")
        ax1.set_title("Max Q(s, g, a) Heatmap")
        fig.colorbar(im1, ax=ax1)
        ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)

        # Policy map with arrows
        ax2.imshow(q_value_map, cmap="gray")
        ax2.set_title("Learned Policy (Arrows)")
        action_arrows = ["↑", "↓", "←", "→", "↖", "↗", "↙", "↘"]
        for r in range(H):
            for c in range(W):
                action = int(policy_map[r, c])
                ax2.text(
                    c,
                    r,
                    action_arrows[action],
                    ha="center",
                    va="center",
                    color="red",
                    fontsize=12,
                )

        plt.suptitle("Policy and Q-value Heatmap")
        if save:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close("all")

    @torch.no_grad()
    def heatmap_subgoal(
        self,
        g_map: torch.Tensor,
        filename: str = "subgoal_heatmap.png",
        save: bool = True,
    ):
        """Visualize the inferred subgoal heatmap with agent/food/wall markers.

        g_map: (1, H, W); filename: where to save the image.
        """
        self.q.eval()
        g_map_np = g_map.squeeze(0).cpu().numpy()  # (H, W)
        agent_pos = self.env.agents[self.learn_ids[0]]
        opponent_pos = self.env.agents[self.hostile_ids[0]]
        food_pos = self.env.food_positions
        wall_pos = self.env.walls

        plt.figure(figsize=(6, 6))
        plt.imshow(g_map_np, cmap="viridis")
        plt.colorbar(label="Inferred Subgoal Probability")
        plt.scatter(
            agent_pos[1], agent_pos[0], color="blue", marker="X", s=100, label="Agent"
        )
        plt.scatter(
            opponent_pos[1],
            opponent_pos[0],
            color="red",
            marker="X",
            s=100,
            label="Opponent",
        )
        if food_pos:
            food_x = [pos[1] for pos in food_pos]
            food_y = [pos[0] for pos in food_pos]
            plt.scatter(food_x, food_y, color="green", marker="o", s=50, label="Food")
        if wall_pos:
            wall_x = [pos[1] for pos in wall_pos]
            wall_y = [pos[0] for pos in wall_pos]
            plt.scatter(wall_x, wall_y, color="black", marker="s", s=50, label="Wall")
        plt.title("Inferred Subgoal Heatmap")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
        if save:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close("all")

    # ------------- acting -------------

    def choose_action(self, qvals: torch.Tensor, beta: float, eval=False) -> int:
        """Gumbel-argmax exploration; eval samples the Boltzmann policy."""
        if eval:
            dist = F.softmax(
                qvals / beta - qvals.max(dim=-1, keepdim=True).values, dim=-1
            )
            return int(torch.multinomial(dist, num_samples=1).item())

        gumbel_noise = -beta * torch.empty_like(qvals).exponential_().log()
        return int(torch.argmax(qvals + gumbel_noise))

    @torch.no_grad()
    def select_action(
        self,
        s_t: np.ndarray,
        s_aug: np.ndarray,
        history: Dict[str, torch.Tensor],
        team_history: Optional[Dict[str, torch.Tensor]] = None,
        eval=False,
    ) -> tuple[int, torch.Tensor, Number]:
        """
        (interaction phase) Infer the hostile and friendly claim maps and act
        eps-greedily on Q(s, g_hostile, g_friendly, *)

        history: cached-feature history dict for the hostile OM (its own
            extractor's features). team_history: same format for the friendly
            OM; defaults to history for single-OM / legacy callers.
        """
        x = torch.from_numpy(s_t).float().unsqueeze(0).to(self.device)
        x_aug = torch.from_numpy(s_aug).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            g_logits = self.model(x, history)  # (1, H, W)
            g_map = torch.sigmoid(g_logits)  # (1, H, W)
            g_team_map = None
            if self.args.friendly_om:
                gt_logits = self.team_model(
                    x, team_history if team_history is not None else history
                )  # (1, H, W)
                g_team_map = torch.sigmoid(gt_logits)  # (1, H, W)

        qvals = self.q(x_aug, g_map, g_team_map)

        tau = self.args.tau_end if eval else self._tau()
        entropy = Categorical(logits=qvals / self.args.tau_end).entropy().item()

        a = self.choose_action(qvals, tau, eval)

        return a, g_map.squeeze(0), entropy

    # ------------- training -------------

    @staticmethod
    def _augment(state: np.ndarray, belief: np.ndarray) -> np.ndarray:
        return np.concatenate([state.astype(np.float32), belief], axis=-1)

    def compute_targets(
        self,
        batch: List[Dict],
        hist_h: Dict[str, torch.Tensor],
        hist_h_nxt: Dict[str, torch.Tensor],
        hist_f: Dict[str, torch.Tensor],
        hist_f_nxt: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Standard DDQN target computation using Hindsight Experience Replay Goal Maps.

        hist_h / hist_h_nxt: hostile-OM cached histories (current / next).
        hist_f / hist_f_nxt: friendly-OM cached histories (current / next).
        """
        s = torch.from_numpy(
            np.array([b["state"] for b in batch], dtype=np.float32)
        ).to(self.device)
        squ = torch.from_numpy(
            np.stack([self._augment(b["state"], b["belief"]) for b in batch])
        ).to(self.device)
        sp = torch.from_numpy(
            np.array([b["next_state"] for b in batch], dtype=np.float32)
        ).to(self.device)
        spu = torch.from_numpy(
            np.stack([self._augment(b["next_state"], b["next_belief"]) for b in batch])
        ).to(self.device)
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
            g_logits = self.model.tgt_model(s, hist_h, cached_features=True)
            g_map = torch.sigmoid(g_logits)  # (B, H, W)

            # Friendly claim maps from the same team-level history (the acting
            # agent's own goal is not part of the label/prediction).
            g_team_map = None
            g_team_map_next = None
            if self.args.friendly_om:
                gt_logits = self.team_model.tgt_model(s, hist_f, cached_features=True)
                g_team_map = torch.sigmoid(gt_logits)  # (B, H, W)

            g_logits_next = self.model.tgt_model(sp, hist_h_nxt, cached_features=True)
            g_map_next = torch.sigmoid(g_logits_next)  # (B, H, W)
            if self.args.friendly_om:
                gt_logits_next = self.team_model.tgt_model(
                    sp, hist_f_nxt, cached_features=True
                )
                g_team_map_next = torch.sigmoid(gt_logits_next)  # (B, H, W)

        # 1. Q(s, g, a)
        q_sa = self.q(squ, g_map, g_team_map).gather(1, a.unsqueeze(1)).squeeze(1)

        # 2. Target = r + gamma * max_a' Q_tgt(s', g, a')
        with torch.no_grad():
            q_val = self.q(spu, g_map_next, g_team_map_next)
            noise = torch.rand_like(q_val) * 1e-6  # tie-breaking noise
            best_actions = (q_val + noise).argmax(dim=1, keepdim=True)

            q_next = (
                self.q_tgt(spu, g_map_next, g_team_map_next)
                .gather(1, best_actions)
                .squeeze(1)
            )

            target = r + (1.0 - done) * self.args.gamma * q_next
            target = torch.clamp(target, min=-15.0, max=15.0)

        return q_sa, target

    def update(self):
        if len(self.replay) < self.args.min_replay:
            return (None, None, None)

        if self.global_step % self.args.train_every != 0:
            return (None, None, None)

        batch_list = self.replay.sample(self.args.batch_size)

        # One shared cache: per-step OM features stored at rollout are collated
        # into the current/next history windows for each OM (no re-embedding).
        h_cache = self.model.collate_cached_history_pair(batch_list, "feats_hostile")
        hist_h, hist_h_nxt = h_cache["cur"], h_cache["nxt"]
        f_cache = self.model.collate_cached_history_pair(batch_list, "feats_friendly")
        hist_f, hist_f_nxt = f_cache["cur"], f_cache["nxt"]

        states = torch.from_numpy(
            np.array([b["state"] for b in batch_list], dtype=np.float32)
        ).to(self.device)

        # OM training batches: hostile + friendly claim-map targets
        om_batch = {
            "states": states,
            "history": hist_h,
            "true_goal_map": torch.from_numpy(
                np.array([b["true_goal_map"] for b in batch_list], dtype=np.float32)
            ).to(self.device),
        }
        team_batch = {
            "states": states,
            "history": hist_f,
            "true_goal_map": torch.from_numpy(
                np.array(
                    [b["true_team_goal_map"] for b in batch_list], dtype=np.float32
                )
            ).to(self.device),
        }

        # Q-network update
        q_sa, target = self.compute_targets(
            batch_list, hist_h, hist_h_nxt, hist_f, hist_f_nxt
        )
        loss = F.smooth_l1_loss(q_sa, target, reduction="mean")
        loss_val = loss.item()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()

        # Soft target update
        with torch.no_grad():
            for param, target_param in zip(
                self.q.parameters(), self.q_tgt.parameters()
            ):
                target_param.lerp_(param, self.args.tau_soft)

        # OM updates (cached features: no re-embedding)
        model_loss = self.model.train_step(om_batch, cached_features=True)
        team_loss = self.team_model.train_step(team_batch, cached_features=True)

        return loss_val, model_loss, team_loss

    def _apply_hindsight_relabeling(
        self,
        episode_transitions: List[Dict],
        step_records: List[Dict],
        final_positions: Dict,
        H: int,
        W: int,
    ):
        """
        Team claim-count labeling from per-agent hindsight subgoals.

        For every agent (friend or foe) the intended subgoal at step t is the
        next goal it collects after t (hindsight); agents that never collect
        are labeled with their final position (the truncated-episode
        heuristic). Per step this yields:
          - true_goal_map:      hostile-team claim map (sum = #hostile agents)
          - true_team_goal_map: teammates' claim map for the acting agent
                                (sum = #teammates; acting agent excluded)
        Maps are claim COUNTS, so a cell can hold > 1 when several agents aim
        at the same goal.
        """
        if not step_records:
            for t in episode_transitions:
                t["true_goal_map"] = np.zeros((H, W), dtype=np.float32)
                t["true_team_goal_map"] = np.zeros((H, W), dtype=np.float32)
            return

        # Shared with offline collection (omexplore.utils.labeling) so the
        # pretraining labels exactly match the RL-time labels.
        agent_goals = compute_agent_goals(step_records, final_positions)

        for tr in episode_transitions:
            goals = agent_goals[tr["step_idx"]]
            tr["true_goal_map"] = claim_count_map(self.hostile_ids, goals, H, W)
            tr["true_team_goal_map"] = claim_count_map(
                [m for m in self.learn_ids if m != tr["agent_id"]], goals, H, W
            )

    # ------------- rollout -------------

    def run_episode(self, opponent_agent, max_steps: int = 500) -> Dict[str, float]:
        """
        Gathers a trajectory for the whole learning team (one shared Q-net),
        controls the hostile teams via opponent_agent (TeamAgent interface:
        reset() + select_actions(obs) -> {agent_id: action}), and labels
        per-team claim maps with hindsight at the end of the episode.
        """
        obs = self.env.reset()
        if random.random() < 0.3:
            obs = self.env.reset_random_spawn()
        elif random.random() < 0.5:
            obs = self.env.swap_agents()
        opponent_agent.reset()
        self.tracker.reset(use_map_prior=self.args.belief_map_prior)
        anchor = self.learn_ids[0]
        self.tracker.update(obs[anchor])

        H, W, _ = obs[anchor].shape

        ep_entropy = 0.0
        ep_shaped = 0.0
        q_losses, model_losses, team_losses = [], [], []

        # History buffers for the transformer (team-level: anchor obs stream;
        # team members' obs differ only in the self channel). One rolling
        # feature window per OM (each has its own CNN extractor).
        history_len = self.args.max_history_length
        rolling_feats = torch.zeros(
            (1, history_len, self.args.d_model), device=self.device
        )
        team_rolling_feats = torch.zeros(
            (1, history_len, self.args.d_model), device=self.device
        )
        rolling_mask = torch.zeros(
            (1, history_len), dtype=torch.bool, device=self.device
        )
        current_seq_len = 0
        prev_state_tensor = torch.zeros((1, *obs[anchor].shape), device=self.device)

        episode_transitions = []
        ep_states = []
        ep_feats_hostile = []
        ep_feats_friendly = []
        step_records = []

        for step in range(max_steps):
            history = {
                "state_features": rolling_feats,
                "mask": rolling_mask,
                "prev_obs": prev_state_tensor,
            }
            team_history = {
                "state_features": team_rolling_feats,
                "mask": rolling_mask,
                "prev_obs": prev_state_tensor,
            }

            # The learning team: one shared Q-net, one action per member.
            actions = {}
            for a in self.learn_ids:
                s_aug = self.tracker.augment(obs[a])
                act, _, step_entropy = self.select_action(
                    obs[a], s_aug, history, team_history
                )
                actions[a] = act
                ep_entropy += step_entropy

            # The hostile team(s), controlled externally.
            opp_actions = opponent_agent.select_actions(obs)
            for a in self.hostile_ids:
                actions[a] = opp_actions[a]

            # Belief state at action time (before this step's update).
            belief_now = self.tracker.channels()

            next_obs, rewards, done, info = self.env.step(actions)
            self.tracker.update(next_obs[anchor])
            next_belief = self.tracker.channels()
            # True optimization return (goal shares + any reward shaping).
            ep_shaped += sum(rewards[a] for a in self.learn_ids)
            for a in self.learn_ids:
                episode_transitions.append(
                    {
                        "agent_id": a,
                        "state": obs[a].copy(),
                        "belief": belief_now.copy(),
                        "action": actions[a],
                        "reward": float(rewards[a]),
                        "next_state": next_obs[a].copy(),
                        "next_belief": next_belief.copy(),
                        "done": bool(done),
                        "hist_len": len(ep_states),
                        "step_idx": step,
                    }
                )
            ep_states.append(obs[anchor].copy())
            step_records.append({"collectors": info.get("collectors", {})})

            # Update history from the anchor's observation stream: cache the
            # feature for this state once per OM (prev = previous anchor state).
            state_tensor = (
                torch.from_numpy(obs[anchor]).float().unsqueeze(0).to(self.device)
            )
            with torch.no_grad():
                new_feat = self.model.inference_model.get_features(
                    state_tensor, prev_state_tensor
                )
                new_team_feat = self.team_model.inference_model.get_features(
                    state_tensor, prev_state_tensor
                )

            ep_feats_hostile.append(new_feat.squeeze(0).cpu().numpy())
            ep_feats_friendly.append(new_team_feat.squeeze(0).cpu().numpy())

            rolling_feats = torch.roll(rolling_feats, shifts=-1, dims=1)
            team_rolling_feats = torch.roll(team_rolling_feats, shifts=-1, dims=1)
            rolling_mask = torch.roll(rolling_mask, shifts=-1, dims=1)
            rolling_feats[:, -1, :] = new_feat
            team_rolling_feats[:, -1, :] = new_team_feat
            if current_seq_len < history_len:
                current_seq_len += 1
            rolling_mask[:, -current_seq_len:] = True

            prev_state_tensor = state_tensor

            # Train step
            self.global_step += 1
            Q_loss, model_loss, team_loss = self.update()
            q_losses.append(Q_loss)
            model_losses.append(model_loss)
            team_losses.append(team_loss)

            obs = next_obs

            if done:
                break

        final_positions = self.env.get_agent_positions()
        self._apply_hindsight_relabeling(
            episode_transitions, step_records, final_positions, H, W
        )

        # Push to replay buffer (shared arrays by reference: states + cached
        # per-step features for both OMs)
        if ep_states:
            states_arr = np.stack(ep_states)
        else:
            states_arr = np.zeros((0, H, W, self.args.state_shape[2]), dtype=np.int8)
        feats_h_arr = (
            np.stack(ep_feats_hostile)
            if ep_feats_hostile
            else np.zeros((0, self.args.d_model), dtype=np.float32)
        )
        feats_f_arr = (
            np.stack(ep_feats_friendly)
            if ep_feats_friendly
            else np.zeros((0, self.args.d_model), dtype=np.float32)
        )
        for t in episode_transitions:
            t["history"] = {
                "states": states_arr,
                "feats_hostile": feats_h_arr,
                "feats_friendly": feats_f_arr,
            }
            self.replay.push(t)

        def _avg(xs):
            xs = [x for x in xs if x is not None]
            return float(np.mean(xs)) if xs else 0.0

        team_score = self.env.team_scores.get(0, 0.0)
        opp_score = sum(s for t, s in self.env.team_scores.items() if t != 0)
        return {
            "return": team_score,
            "steps": step + 1,
            "opp_return": opp_score,
            "shaped_return": ep_shaped,
            "avg_entropy": ep_entropy / max(1, (step + 1) * len(self.learn_ids)),
            "avg_q_loss": _avg(q_losses),
            "avg_model_loss": _avg(model_losses),
            "avg_team_model_loss": _avg(team_losses),
        }

    def run_test_episode(
        self, opponent_agent, max_steps: int = 500, render: bool = False
    ) -> Dict[str, float]:
        """
        Evaluation rollout: no exploration noise schedule, no replay /
        training. Reports the hostile OM's prediction quality (target MAE /
        spatial error) against the opponent team's true claim heatmap whenever
        the opponents have a known target. `render` is accepted for API
        compatibility; a team-env renderer is still TBD.
        """
        self.model.inference_model.eval()
        self.team_model.inference_model.eval()
        obs = self.env.reset()
        opponent_agent.reset()
        self.tracker.reset(use_map_prior=self.args.belief_map_prior)
        anchor = self.learn_ids[0]
        self.tracker.update(obs[anchor])

        ep_entropy = 0.0
        ep_shaped = 0.0
        ep_mae_errors = []
        ep_spatial_errors = []

        history_len = self.args.max_history_length
        rolling_feats = torch.zeros(
            (1, history_len, self.args.d_model), device=self.device
        )
        team_rolling_feats = torch.zeros(
            (1, history_len, self.args.d_model), device=self.device
        )
        rolling_mask = torch.zeros(
            (1, history_len), dtype=torch.bool, device=self.device
        )
        current_seq_len = 0
        prev_state_tensor = torch.zeros((1, *obs[anchor].shape), device=self.device)

        for step in range(max_steps):
            history = {
                "state_features": rolling_feats,
                "mask": rolling_mask,
                "prev_obs": prev_state_tensor,
            }
            team_history = {
                "state_features": team_rolling_feats,
                "mask": rolling_mask,
                "prev_obs": prev_state_tensor,
            }

            actions = {}
            g_map_anchor = None
            for a in self.learn_ids:
                s_aug = self.tracker.augment(obs[a])
                act, g_map, step_entropy = self.select_action(
                    obs[a], s_aug, history, team_history, eval=True
                )
                actions[a] = act
                ep_entropy += step_entropy
                if a == anchor:
                    g_map_anchor = g_map

            opp_actions = opponent_agent.select_actions(obs)
            for a in self.hostile_ids:
                actions[a] = opp_actions[a]

            # OM prediction quality vs the hostile team's true claims
            # (per-cell probabilities vs the binarized claim map).
            opp_heat = opponent_agent.get_team_heatmap()
            if g_map_anchor is not None and opp_heat is not None:
                if opp_heat.sum() > 0:
                    g2 = (
                        g_map_anchor.unsqueeze(0)
                        if g_map_anchor.dim() == 2
                        else g_map_anchor
                    )
                    tgt = torch.from_numpy(opp_heat).to(self.device).unsqueeze(0)
                    ep_mae_errors.append(self.model.heatmap_mae(g2, tgt))
                    ep_spatial_errors.append(self.model.expected_spatial_error(g2, tgt))

            next_obs, rewards, done, info = self.env.step(actions)
            self.tracker.update(next_obs[anchor])
            ep_shaped += sum(rewards[a] for a in self.learn_ids)

            state_tensor = (
                torch.from_numpy(obs[anchor]).float().unsqueeze(0).to(self.device)
            )
            with torch.no_grad():
                new_feat = self.model.inference_model.get_features(
                    state_tensor, prev_state_tensor
                )
                new_team_feat = self.team_model.inference_model.get_features(
                    state_tensor, prev_state_tensor
                )

            rolling_feats = torch.roll(rolling_feats, shifts=-1, dims=1)
            team_rolling_feats = torch.roll(team_rolling_feats, shifts=-1, dims=1)
            rolling_mask = torch.roll(rolling_mask, shifts=-1, dims=1)

            rolling_feats[:, -1, :] = new_feat
            team_rolling_feats[:, -1, :] = new_team_feat

            if current_seq_len < history_len:
                current_seq_len += 1
            rolling_mask[:, -current_seq_len:] = True

            prev_state_tensor = state_tensor
            obs = next_obs

            if done:
                break

        team_score = self.env.team_scores.get(0, 0.0)
        opp_score = sum(s for t, s in self.env.team_scores.items() if t != 0)
        return {
            "return": team_score,
            "steps": step + 1,
            "opp_return": opp_score,
            "shaped_return": ep_shaped,
            "avg_entropy": ep_entropy / max(1, (step + 1) * len(self.learn_ids)),
            "avg_mae_error": float(np.mean(ep_mae_errors)) if ep_mae_errors else None,
            "avg_spatial_error": (
                float(np.mean(ep_spatial_errors)) if ep_spatial_errors else None
            ),
        }

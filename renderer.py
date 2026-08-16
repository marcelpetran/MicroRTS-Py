# matplotlib.use("TkAgg")
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class RealtimeRenderer:
    def __init__(self):
        # We need Matplotlib for drawing. We use interactive mode.
        plt.ion()
        self.fig, self.axs = plt.subplots(1, 4, figsize=(20, 5))
        self.im_global = None
        self.im_p1 = None
        self.im_p2 = None
        self.im_om = None

    def _obs_to_rgb(self, obs):
        H, W, _ = obs.shape
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        has_fog = obs.shape[2] >= 6
        for r in range(H):
            for c in range(W):
                if has_fog and obs[r, c, 5] == 0:
                    rgb[r, c] = [0.2, 0.2, 0.2]  # Dark gray fog
                elif obs[r, c, 4] == 1:
                    rgb[r, c] = [0, 0, 0]  # Black wall
                elif obs[r, c, 1] == 1:
                    rgb[r, c] = [0, 1, 0]  # Green food
                elif obs[r, c, 2] == 1 and obs[r, c, 3] == 1:
                    rgb[r, c] = [1, 0, 1]  # Magenta both agents
                elif obs[r, c, 2] == 1:
                    rgb[r, c] = [0, 0, 1]  # Blue Agent 0 (Self)
                elif obs[r, c, 3] == 1:
                    rgb[r, c] = [1, 0, 0]  # Red Agent 1 (Opponent)
                else:
                    rgb[r, c] = [1, 1, 1]  # White floor
        return rgb

    def render(self, global_state, obs0, obs1, om_pred):
        rgb_global = self._obs_to_rgb(global_state)
        rgb_0 = self._obs_to_rgb(obs0)
        rgb_1 = self._obs_to_rgb(obs1)

        # om_pred could be a torch Tensor or numpy array
        if hasattr(om_pred, "detach"):
            om_pred = om_pred.detach().cpu().numpy()

        # Squeeze in case it has extra batch or channel dimensions
        om_pred = np.squeeze(om_pred)

        if self.im_global is None:
            self.im_global = self.axs[0].imshow(rgb_global)
            self.axs[0].set_title("Global State")
            self.axs[0].axis("off")

            self.im_p1 = self.axs[1].imshow(rgb_0)
            self.axs[1].set_title("Agent 0 (Blue) View")
            self.axs[1].axis("off")

            self.im_p2 = self.axs[2].imshow(rgb_1)
            self.axs[2].set_title("Agent 1 (Red) View")
            self.axs[2].axis("off")

            self.im_om = self.axs[3].imshow(
                om_pred, cmap="hot", interpolation="nearest", vmin=0, vmax=1
            )
            self.axs[3].set_title("OM Predictions (Agent 0 Inference)")
            self.axs[3].axis("off")
            self.fig.colorbar(self.im_om, ax=self.axs[3], fraction=0.046, pad=0.04)
        else:
            self.im_global.set_data(rgb_global)
            self.im_p1.set_data(rgb_0)
            self.im_p2.set_data(rgb_1)
            self.im_om.set_data(om_pred)

        # Allow matplotlib to flush the GUI events
        plt.pause(0.1)

    def close(self):
        plt.ioff()
        plt.close(self.fig)

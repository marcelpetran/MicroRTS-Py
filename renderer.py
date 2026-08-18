import matplotlib

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


class RealtimeRenderer:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(16, 6), facecolor="#1a1a2e")
        self.fig.suptitle(
            "Foraging Environment", color="white", fontsize=16, fontweight="bold"
        )

        # Modern color scheme
        self.colors = {
            "bg": "#1a1a2e",
            "fog": "#2d2d44",
            "wall": "#4a4a6a",
            "food": "#00d9a5",
            "agent0": "#4cc9f0",
            "agent1": "#f72585",
            "both": "#b5179e",
            "text": "white",
        }

        # Create custom colormap for OM predictions
        self.om_cmap = LinearSegmentedColormap.from_list(
            "modern_hot", ["#1a1a2e", "#f72585", "#ffb703", "#ffffff"], N=256
        )

        self.axs = []
        self.ims = []
        self._init_subplots()

    def _init_subplots(self):
        titles = [
            "Global View",
            "Agent 0 Perspective",
            "Agent 1 Perspective",
            "Opponent Model",
        ]

        for i in range(4):
            ax = self.fig.add_subplot(1, 4, i + 1)
            ax.set_facecolor(self.colors["bg"])
            ax.set_title(titles[i], color=self.colors["text"], fontsize=12, pad=10)
            ax.axis("off")
            self.axs.append(ax)
            self.ims.append(None)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

    def _obs_to_rgb(self, obs):
        """Vectorized RGB conversion with modern color palette."""
        H, W, _ = obs.shape
        rgb = np.ones((H, W, 3), dtype=np.float32) * 0.1  # Dark bg

        has_fog = obs.shape[2] >= 6

        # Fog of war
        if has_fog:
            mask_fog = obs[..., 5] == 0
            rgb[mask_fog] = [0.18, 0.18, 0.27]

        # Walls
        mask_wall = obs[..., 4] == 1
        rgb[mask_wall] = [0.29, 0.29, 0.42]

        # Food (bright teal)
        mask_food = obs[..., 1] == 1
        rgb[mask_food] = [0.0, 0.85, 0.65]

        # Agent 0 (cyan)
        mask_a0 = obs[..., 2] == 1
        rgb[mask_a0] = [0.29, 0.79, 0.95]

        # Agent 1 (magenta)
        mask_a1 = obs[..., 3] == 1
        rgb[mask_a1] = [0.97, 0.15, 0.52]

        # Both agents (purple)
        mask_both = (obs[..., 2] == 1) & (obs[..., 3] == 1)
        rgb[mask_both] = [0.71, 0.09, 0.62]

        return rgb

    def render(self, global_state, obs0, obs1, om_pred):
        rgb_global = self._obs_to_rgb(global_state)
        rgb_0 = self._obs_to_rgb(obs0)
        rgb_1 = self._obs_to_rgb(obs1)

        # Process OM prediction
        om_pred = np.squeeze(om_pred)
        if hasattr(om_pred, "detach"):
            om_pred = om_pred.detach().cpu().numpy()

        # Initialize or update
        if self.ims[0] is None:
            self.ims[0] = self.axs[0].imshow(rgb_global, interpolation="nearest")
            self.ims[1] = self.axs[1].imshow(rgb_0, interpolation="nearest")
            self.ims[2] = self.axs[2].imshow(rgb_1, interpolation="nearest")
            self.ims[3] = self.axs[3].imshow(
                om_pred,
                cmap=self.om_cmap,
                interpolation="nearest",
                vmin=0,
                vmax=1,
                alpha=0.9,
            )
            # Add colorbar for OM
            cbar = self.fig.colorbar(
                self.ims[3], ax=self.axs[3], fraction=0.046, pad=0.04
            )
            cbar.ax.tick_params(colors=self.colors["text"])
        else:
            self.ims[0].set_data(rgb_global)
            self.ims[1].set_data(rgb_0)
            self.ims[2].set_data(rgb_1)
            self.ims[3].set_data(om_pred)

        plt.pause(0.05)

    def close(self):
        plt.ioff()
        plt.close(self.fig)

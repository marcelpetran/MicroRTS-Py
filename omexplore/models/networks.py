import torch
from torch import nn

from omexplore.utils.omg_args import OMGArgs


class QNet(nn.Module):
    """
    RL Network Q(s, g, a) that learns best response with imbued heatmaps of
    opponent and teammate subgoal inference.
    state_shape: (H, W, F)
    action_dim: number of discrete actions
    g_map: (B, H, W) heatmap of inferred hostile subgoals
    g_team_map: (B, H, W) heatmap of inferred teammate subgoals (optional,
        only used when args.friendly_om is set; zeros are fed otherwise)
    output: Q-values for each action
    Dueling architecture with shared CNN backbone and separate value/advantage heads.
    """

    def __init__(self, args: OMGArgs):
        super().__init__()
        H, W, F_dim = args.state_shape
        self.state_dim: int = H * W * F_dim
        self.action_dim: int = args.action_dim
        self.friendly_om: bool = getattr(args, "friendly_om", True)
        cnn_hidden = args.cnn_hidden

        self.flat_dim: int = cnn_hidden * H * W
        # +1 hostile subgoal heatmap, +1 friendly subgoal heatmap (ablatable)
        input_channels = F_dim + args.belief_channels + (2 if self.friendly_om else 1)

        self.cnn: nn.Sequential = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, cnn_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_hidden, cnn_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Heads (Dueling)
        self.advantage_head: nn.Sequential = nn.Sequential(
            nn.Linear(self.flat_dim, args.qnet_hidden),
            nn.ReLU(),
            nn.Linear(args.qnet_hidden, self.action_dim),
        )

        self.value_head: nn.Sequential = nn.Sequential(
            nn.Linear(self.flat_dim, args.qnet_hidden),
            nn.ReLU(),
            nn.Linear(args.qnet_hidden, 1),
        )

        # Initialize weights to small values to prevent initial explosion
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)

    def forward(
        self, batch: torch.Tensor, g_map: torch.Tensor, g_team_map: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            batch: Game state (B, H, W, F)  [F already includes belief channels]
            g_map: Heatmap of inferred hostile subgoals (B, H, W)
            g_team_map: Heatmap of inferred teammate subgoals (B, H, W); only
                used when constructed with friendly_om=True (zeros otherwise)
        """
        # Batch shape: (B, H, W, F) -> (B, F, H, W) for PyTorch Conv2d
        s = batch.permute(0, 3, 1, 2)
        # (B, 1, H, W) - broadcast heatmaps across spatial dimensions
        chans = [s, g_map.unsqueeze(1)]
        if self.friendly_om:
            if g_team_map is None:
                g_team_map = torch.zeros_like(g_map)
            chans.append(g_team_map.unsqueeze(1))
        x = torch.cat(chans, dim=1)
        features = self.cnn(x)

        # Dueling Heads
        adv = self.advantage_head(features)
        val = self.value_head(features)
        q_vals = val + adv - adv.mean(dim=1, keepdim=True)

        return q_vals


class QNetClassic(nn.Module):
    """
    RL Network: Q(s, a)
    Learns the Best Response to the opponent's average strategy.
    """

    def __init__(self, args: OMGArgs):
        super().__init__()
        H, W, F_dim = args.state_shape
        F_dim += args.belief_channels
        self.state_dim = H * W * F_dim
        self.action_dim = args.action_dim
        cnn_hidden = args.cnn_hidden
        self.flat_dim = cnn_hidden * H * W

        self.cnn = nn.Sequential(
            nn.Conv2d(F_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, cnn_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_hidden, cnn_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Heads (Dueling)
        self.advantage_head = nn.Sequential(
            nn.Linear(self.flat_dim, args.qnet_hidden),
            nn.ReLU(),
            nn.Linear(args.qnet_hidden, self.action_dim),
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.flat_dim, args.qnet_hidden),
            nn.ReLU(),
            nn.Linear(args.qnet_hidden, 1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # Batch shape: (B, H, W, F) -> Permute to (B, F, H, W) for Conv2d
        s = batch.permute(0, 3, 1, 2)
        # x = torch.cat([s[:, :3], s[:, 4:]], dim=1)
        features = self.cnn(s)

        # Dueling Heads
        adv = self.advantage_head(features)
        val = self.value_head(features)
        q_vals = val + adv - adv.mean(dim=1, keepdim=True)

        return q_vals


class SLnet(nn.Module):
    """
    SL Network: Pi(a | s)
    Learns the agent's own average historical strategy.
    """

    def __init__(self, args: OMGArgs):
        super().__init__()
        H, W, F_dim = args.state_shape
        self.state_dim = H * W * F_dim
        self.action_dim = args.action_dim
        cnn_hidden = args.cnn_hidden
        self.flat_dim = cnn_hidden * H * W

        self.cnn = nn.Sequential(
            nn.Conv2d(F_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, cnn_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_hidden, cnn_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.flat_dim, args.qnet_hidden),
            nn.ReLU(),
            nn.Linear(args.qnet_hidden, self.action_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        s = batch.permute(0, 3, 1, 2)
        features = self.cnn(s)
        logits = self.value_head(features)
        return logits

import torch
from torch import nn

from omexplore.models.graph_features import GLOBAL_FEATURES, N_NODE_FEATURES
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

    @staticmethod
    def _head_forward(head: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 1:
            x = torch.addmv(head[0].bias, head[0].weight, x[0]).unsqueeze(0)
            return head[1:](x)
        return head(x)

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
        adv = self._head_forward(self.advantage_head, features)
        val = self._head_forward(self.value_head, features)
        q_vals = val + adv - adv.mean(dim=1, keepdim=True)

        return q_vals


class QNetTemporal(nn.Module):
    """Q(s, g, a) with a map-wide receptive field (temporal agent).

    Drop-in replacement for QNet: same forward signature
    (batch (B, H, W, F'), g_map (B, H, W), g_team_map (B, H, W)) -> (B, A),
    built for the large MovingAI maps where QNet's 3x(3x3) backbone
    (7x7 receptive field) cannot relate distant believed goals / OM
    heatmap cells to the acting agent's position.

    Differences vs QNet:
    - Backbone: dilated 3x3 conv blocks (dilation 1..32, receptive field
      ~253 cells) with GroupNorm (no batch statistics: eval-safe and
      uncorrelated with TD targets, unlike BatchNorm).
    - Readout: features at the SELF cell (gathered via the input's self
      channel) + global avg/max pool, then dueling heads. Spatially
      shared and ~1000x fewer parameters than QNet's Flatten->Linear
      heads (~345M on den312d with qnet_hidden=512).
    """

    def __init__(self, args: OMGArgs):
        super().__init__()
        H, W, F_dim = args.state_shape
        self.action_dim: int = args.action_dim
        self.friendly_om: bool = getattr(args, "friendly_om", True)
        c = args.cnn_hidden
        groups = 8 if c % 8 == 0 else 1
        input_channels = F_dim + args.belief_channels + (2 if self.friendly_om else 1)

        blocks: list[nn.Module] = []
        in_ch = input_channels
        for d in (1, 2, 4, 8, 16, 32):
            blocks += [
                nn.Conv2d(in_ch, c, 3, padding=d, dilation=d),
                nn.GroupNorm(groups, c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=d, dilation=d),
                nn.GroupNorm(groups, c),
                nn.ReLU(inplace=True),
            ]
            in_ch = c
        self.cnn = nn.Sequential(*blocks)

        head_in = 3 * c  # self-cell features + global avg + global max
        self.advantage_head = nn.Sequential(
            nn.Linear(head_in, args.qnet_hidden),
            nn.ReLU(),
            nn.Linear(args.qnet_hidden, self.action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(head_in, args.qnet_hidden),
            nn.ReLU(),
            nn.Linear(args.qnet_hidden, 1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)

    def forward(
        self, batch: torch.Tensor, g_map: torch.Tensor, g_team_map: torch.Tensor = None
    ) -> torch.Tensor:
        s = batch.permute(0, 3, 1, 2)
        chans = [s, g_map.unsqueeze(1)]
        if self.friendly_om:
            if g_team_map is None:
                g_team_map = torch.zeros_like(g_map)
            chans.append(g_team_map.unsqueeze(1))
        feats = self.cnn(torch.cat(chans, dim=1))  # (B, c, H, W)

        b, c, h, w = feats.shape
        flat = feats.reshape(b, c, h * w)
        # Features at the acting agent's cell (input state channel 2 = self).
        self_pos = batch[:, :, :, 2].reshape(b, h * w).argmax(dim=1)  # (B,)
        idx = self_pos.view(b, 1, 1).expand(b, c, 1)
        f_self = flat.gather(2, idx).squeeze(2)  # (B, c)
        f_avg = feats.mean(dim=(2, 3))
        f_max = feats.amax(dim=(2, 3))
        h_vec = torch.cat([f_self, f_avg, f_max], dim=1)  # (B, 3c)

        adv = self.advantage_head(h_vec)
        val = self.value_head(h_vec)
        return val + adv - adv.mean(dim=1, keepdim=True)


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


class Structure2Vec(nn.Module):
    """
    Message-passing backbone for graph neural networks, that embeds the graph into a vector representation.
    """

    def __init__(self, args: OMGArgs):
        super().__init__()

    def forward(x, edge_index, edge_weight):
        # → μ (B, N, s2v_dim)
        raise NotImplementedError("Structure2Vec forward pass not implemented.")


class QNetGraph(nn.Module):
    """Owns a Structure2Vec plus the dueling heads."""

    def __init__(self, args: OMGArgs):
        super().__init__()
        self.s2v = Structure2Vec(args)
        self.advantage_head = ...
        self.value_head = ...

    def forward(x, edge_index, edge_weight, g):
        # → (B, N)
        raise NotImplementedError("GNNet forward pass not implemented.")

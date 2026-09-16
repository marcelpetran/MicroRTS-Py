"""Episode renderer for TeamRoadmapEnv — thesis showcase visuals.

Renders a recorded episode as a GIF (+ key PNG frames), or live in an
interactive window (LiveRenderer):

  Main panel   — the map as seen by the learning team (team 0): walls,
                 team-pooled fog of war, explored-but-not-visible tint,
                 visible goals, agents (team 0 solid, visible opponents
                 solid, opponents in fog shown as faint "ghosts").
  Side panels  — hostile / friendly claim maps:
                   * OM prediction (what the model infers from obs)
                   * true intent  (oracle: the scripted agent's targets)

The episode is first recorded into lightweight per-step snapshots
(`snapshot`), then animated (`render_episode`) with
matplotlib.animation + PillowWriter, so no interactive backend is needed.

Color scheme follows the legacy RealtimeRenderer (dark #1a1a2e theme,
team 0 cyan #4cc9f0, team 1 magenta #f72585, goals teal #00d9a5).
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.animation
import numpy as np

try:
    from matplotlib.colors import LinearSegmentedColormap
except Exception:  # pragma: no cover - matplotlib always present in practice
    LinearSegmentedColormap = None

# ----------------------------------------------------------------------
# Palette (kept in sync with utils/renderer.py)
COL = {
    "bg": "#101020",  # out-of-everything
    "fog": "#2d2d44",  # never seen by team 0
    "seen": "#22223a",  # explored, not currently visible
    "vis": "#1a1a2e",  # currently in team-0 vision
    "wall": "#4a4a6a",
    "goal": "#00d9a5",
    "goals": "#005b46",  # goal in fog
    "team0": "#4cc9f0",
    "team1": "#f72585",
    "text": "#ffffff",
}

RGB = {
    k: tuple(int(v[i : i + 2], 16) / 255.0 for i in (1, 3, 5)) for k, v in COL.items()
}

if LinearSegmentedColormap is not None:
    HOSTILE_CMAP = LinearSegmentedColormap.from_list(
        "om_hostile", ["#1a1a2e", "#f72585", "#ffb703", "#ffffff"], N=256
    )
    FRIENDLY_CMAP = LinearSegmentedColormap.from_list(
        "om_friendly", ["#1a1a2e", "#4cc9f0", "#ffb703", "#ffffff"], N=256
    )
else:  # pragma: no cover
    HOSTILE_CMAP = FRIENDLY_CMAP = None


# ----------------------------------------------------------------------
# Recording
# ----------------------------------------------------------------------
@dataclass
class FrameData:
    step: int
    rgb: np.ndarray  # (H, W, 3) float32 background (no agents)
    agents: list  # [(r, c, team, visible_to_team0), ...]
    scores: dict  # team_id -> score
    hostile_true: np.ndarray | None = None
    friendly_true: np.ndarray | None = None
    hostile_pred: np.ndarray | None = None
    friendly_pred: np.ndarray | None = None


def _to_np(m):
    if m is None:
        return None
    if hasattr(m, "detach"):
        m = m.detach().cpu().numpy()
    return np.asarray(m, dtype=np.float32).squeeze()


def snapshot(
    env,
    step: int,
    hostile_true=None,
    friendly_true=None,
    hostile_pred=None,
    friendly_pred=None,
) -> FrameData:
    """One per-step snapshot from the env (oracle positions + team-0 fog)."""
    H, W = env.height, env.width
    vis = np.asarray(env._team_vis[0], dtype=bool)
    coverage = np.asarray(env._coverage[0], dtype=bool)
    walls = np.zeros((H, W), dtype=bool)
    walls[env._wall_rows, env._wall_cols] = True

    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[:] = RGB["bg"]
    rgb[coverage] = RGB["seen"]
    rgb[vis] = RGB["vis"]
    rgb[walls] = RGB["wall"]
    for r, c in env.food_positions:
        if vis[r, c]:
            rgb[r, c] = RGB["goal"]
        else:
            rgb[r, c] = RGB["goals"]

    agents = []
    for a, (r, c) in env.get_agent_positions().items():
        t = env.teams[a]
        agents.append((r, c, t, bool(vis[r, c])))

    return FrameData(
        step=step,
        rgb=rgb,
        agents=agents,
        scores=dict(env.team_scores),
        hostile_true=_to_np(hostile_true),
        friendly_true=_to_np(friendly_true),
        hostile_pred=_to_np(hostile_pred),
        friendly_pred=_to_np(friendly_pred),
    )


# ----------------------------------------------------------------------
# Animation
# ----------------------------------------------------------------------
def _panel_vmax(frames, attr):
    m = 0.0
    for f in frames:
        a = getattr(f, attr)
        if a is not None:
            m = max(m, float(a.max()) if a.size else 0.0)
    return m if m > 1e-9 else 1.0


class _EpisodeFigure:
    """Episode figure (main map + claim-map panels) shared by the GIF and
    live rendering paths."""

    PANEL_SPECS = (
        ("hostile_pred", "Hostile claims — OM prediction", HOSTILE_CMAP),
        ("hostile_true", "Hostile claims — true (oracle)", HOSTILE_CMAP),
        ("friendly_pred", "Friendly claims — OM prediction", FRIENDLY_CMAP),
        ("friendly_true", "Friendly claims — true (oracle)", FRIENDLY_CMAP),
    )

    def __init__(
        self,
        panel_keys,
        H,
        W,
        title="Team-based competitive exploration",
        team_names=("Team 0", "Team 1"),
        dpi=100,
        first_frame=None,
        vmax=None,
    ):
        """panel_keys: FrameData fields to show as side panels (in
        PANEL_SPECS order). vmax: optional {key: color-scale max}."""
        import matplotlib.pyplot as plt

        self._plt = plt
        self.team_names = team_names
        self.H, self.W = H, W
        vmax = vmax or {}

        specs = [
            (key, ptitle, cmap, vmax.get(key, 1.0))
            for key, ptitle, cmap in self.PANEL_SPECS
            if key in panel_keys
        ]
        n_panels = len(specs)

        # Layout: main map on the left, panels in a ceil(n/2) x 2 grid on the right.
        n_rows = max(1, (n_panels + 1) // 2)
        fig = plt.figure(
            figsize=(6.5 + 3.2 * min(n_panels, 2), 6.0), facecolor=COL["bg"], dpi=dpi
        )
        gs = fig.add_gridspec(
            n_rows,
            1 + min(n_panels, 2),
            width_ratios=[1.4] + [1.0] * min(n_panels, 2),
            wspace=0.12,
            hspace=0.25,
            left=0.03,
            right=0.98,
            top=0.90,
            bottom=0.04,
        )

        ax_main = fig.add_subplot(gs[:, 0])
        ax_main.set_facecolor(COL["bg"])
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        self.title_main = ax_main.set_title("", color=COL["text"], fontsize=11)

        init_rgb = first_frame.rgb if first_frame is not None else np.zeros((H, W, 3))
        self.im_main = ax_main.imshow(init_rgb, interpolation="nearest")
        ax_main.set_xlim(-0.5, W - 0.5)
        ax_main.set_ylim(H - 0.5, -0.5)
        ax_main.set_aspect("equal")

        self.sc_own = ax_main.scatter(
            [],
            [],
            s=42,
            c=COL["team0"],
            marker="o",
            edgecolors="white",
            linewidths=0.6,
            zorder=5,
        )
        self.sc_opp = ax_main.scatter(
            [],
            [],
            s=42,
            c=COL["team1"],
            marker="s",
            edgecolors="white",
            linewidths=0.6,
            zorder=5,
        )
        self.sc_ghost = ax_main.scatter(
            [],
            [],
            s=30,
            c=COL["team1"],
            marker="s",
            alpha=0.30,
            linewidths=0.0,
            zorder=4,
        )
        # legend
        ax_main.scatter(
            [],
            [],
            s=42,
            c=COL["team0"],
            marker="o",
            edgecolors="white",
            linewidths=0.6,
            label=f"{team_names[0]}",
        )
        ax_main.scatter(
            [],
            [],
            s=42,
            c=COL["team1"],
            marker="s",
            edgecolors="white",
            linewidths=0.6,
            label=f"{team_names[1]}",
        )
        ax_main.scatter([], [], s=42, c=COL["goal"], marker="s", label="goals")
        ax_main.legend(
            loc="lower left",
            fontsize=7,
            framealpha=0.25,
            facecolor=COL["bg"],
            edgecolor="#4a4a6a",
            labelcolor="white",
        )

        self.im_panels = []
        for i, (key, ptitle, cmap, v) in enumerate(specs):
            row, col = divmod(i, 2)
            ax = fig.add_subplot(gs[row, 1 + col])
            ax.set_facecolor(COL["bg"])
            ax.set_title(ptitle, color=COL["text"], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            data = getattr(first_frame, key, None) if first_frame is not None else None
            data = data if data is not None else np.zeros((H, W), dtype=np.float32)
            im = ax.imshow(data, cmap=cmap, interpolation="nearest", vmin=0.0, vmax=v)
            self.im_panels.append((im, key))

        fig.suptitle(title, color=COL["text"], fontsize=13, fontweight="bold")
        self.fig = fig

    def _coords(self, agents, team, visible):
        pts = [(c, r) for r, c, t, vis in agents if t == team and vis == visible]
        return np.asarray(pts, dtype=float).reshape(-1, 2)

    def update(self, frame):
        self.im_main.set_data(frame.rgb)
        self.sc_own.set_offsets(self._coords(frame.agents, 0, True))
        self.sc_opp.set_offsets(self._coords(frame.agents, 1, True))
        self.sc_ghost.set_offsets(self._coords(frame.agents, 1, False))
        scores = " | ".join(
            f"{self.team_names[t] if t < len(self.team_names) else f'Team {t}'}: {s:g}"
            for t, s in sorted(frame.scores.items())
        )
        self.title_main.set_text(f"step {frame.step}   {scores}")
        for im, key in self.im_panels:
            data = getattr(frame, key)
            im.set_data(data if data is not None else np.zeros((self.H, self.W)))
        return [self.im_main, self.title_main]

    def set_vmax(self, key, vmax):
        for im, k in self.im_panels:
            if k == key:
                im.set_clim(0.0, vmax)

    def close(self):
        self._plt.close(self.fig)


class LiveRenderer:
    """Interactive renderer: show frames as the episode is produced.

    The figure is built lazily from the first frame (side panels follow
    whichever claim maps that frame carries). Panel color scales start at
    the first frame's max and only grow (set_clim), so colors stay stable.

    Typical use:
        live = LiveRenderer(title=...)
        try:
            for frame in rollout(...):
                if live.is_closed:
                    break
                live.show(frame, delay=1.0 / fps)
        except KeyboardInterrupt:
            ...
    """

    def __init__(
        self,
        title="Team-based competitive exploration",
        team_names=("Team 0", "Team 1"),
        dpi=100,
    ):
        self.title = title
        self.team_names = team_names
        self.dpi = dpi
        self._ef = None
        self._vmax = {}

    def show(self, frame, delay=0.15):
        """Draw one frame; `delay` seconds of event processing afterwards."""
        import matplotlib.pyplot as plt

        self._plt = plt
        if self._ef is None:
            H, W = frame.rgb.shape[:2]
            panel_keys = [
                k
                for k, _, _ in _EpisodeFigure.PANEL_SPECS
                if getattr(frame, k) is not None
            ]
            for k in panel_keys:
                data = np.asarray(getattr(frame, k))
                self._vmax[k] = max(float(data.max()) if data.size else 0.0, 1e-6)
            self._ef = _EpisodeFigure(
                panel_keys,
                H,
                W,
                title=self.title,
                team_names=self.team_names,
                dpi=self.dpi,
                first_frame=frame,
                vmax=dict(self._vmax),
            )
        else:
            for k in self._vmax:
                data = getattr(frame, k)
                if data is not None and data.size:
                    m = float(np.asarray(data).max())
                    if m > self._vmax[k]:
                        self._vmax[k] = m
                        self._ef.set_vmax(k, m)
        self._ef.update(frame)
        plt.pause(delay)

    @property
    def is_closed(self):
        """True once the user closed the window (no figure yet -> False)."""
        if self._ef is None:
            return False
        return not self._plt.fignum_exists(self._ef.fig.number)

    def close(self):
        if self._ef is not None:
            self._ef.close()


def render_episode(
    frames: list[FrameData],
    out_path: str | Path,
    fps: int = 6,
    title: str = "Team-based competitive exploration",
    team_names: tuple = ("Team 0", "Team 1"),
    dpi: int = 100,
    keep_pngs: int = 3,
):
    """Animate recorded frames and save a GIF (+ a few PNG key frames)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    panel_keys = []
    vmax = {}
    for key, _, _ in _EpisodeFigure.PANEL_SPECS:
        if any(getattr(f, key) is not None for f in frames):
            panel_keys.append(key)
            vmax[key] = _panel_vmax(frames, key)

    H, W = frames[0].rgb.shape[:2]
    ef = _EpisodeFigure(
        panel_keys,
        H,
        W,
        title=title,
        team_names=team_names,
        dpi=dpi,
        first_frame=frames[0],
        vmax=vmax,
    )

    anim = matplotlib.animation.FuncAnimation(
        ef.fig,
        lambda i: ef.update(frames[i]),
        frames=len(frames),
        interval=int(1000 / fps),
        blit=False,
    )
    writer = matplotlib.animation.PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer)

    # Key PNG frames (start / middle / end), saved before closing the figure.
    png_paths = []
    if keep_pngs > 0 and len(frames) > 0:
        idxs = sorted({0, len(frames) // 2, len(frames) - 1})[:keep_pngs]
        for i in idxs:
            ef.update(frames[i])
    ef.close()
    return out_path, png_paths

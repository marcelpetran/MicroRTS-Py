import numpy as np


class BeliefTracker:
    """
    Belief channels derived ONLY from the observing side's observation
    history. Update rules mirror the heuristic opponents' beliefs:
      - food: add where food observed, discard where cell observed empty [6]
      - opponents: cells where opponents were last seen, cleared when a cell
        is later observed visible-and-empty (they have left); opp_age keeps
        growing so the NN still gets a staleness signal after the drop [6]

    Works for any number of observed opponents (team env: all foes pooled in
    the OPP channel; 1v1 env: reduces to the old single last-seen position).

    Channels (H, W, 3):
      0: belief_food  cells believed to contain food
      1: belief_opp   1.0 at opponents' last observed cells
      2: opp_age      steps since any opponent was last seen, normalized [0, 1]

    `channels` selects the (FOOD, OPP, VIS) channel indices of the obs the
    tracker consumes: (1, 3, 5) for the 6-channel 1v1 obs, (1, 4, 6) for the
    7-channel team obs (goal / opponent / team-vision mask).
    """

    def __init__(self, height, width, map_layout=None, horizon=50, channels=(1, 3, 5)):
        self.H, self.W = height, width
        self.horizon = horizon
        self.FOOD_CH, self.OPP_CH, self.VIS_CH = channels
        self.prior_food, self.prior_opp = set(), set()
        if map_layout is not None:  # same 'o'/'B' static prior heuristics get [6]
            for r, row in enumerate(map_layout):
                for c, ch in enumerate(row):
                    if ch == "o":
                        self.prior_food.add((r, c))
                    elif ch == "B":  # learner is agent 0 -> opponents start at 'B'
                        self.prior_opp.add((r, c))
        self.reset()

    def reset(self, use_map_prior=True):
        self.belief_food = set(self.prior_food) if use_map_prior else set()
        self.belief_opp = set(self.prior_opp) if use_map_prior else set()
        self.opp_age = 0

    def update(self, obs):
        vis = obs[:, :, self.VIS_CH]
        for r, c in np.argwhere(vis == 1):
            if obs[r, c, self.FOOD_CH] == 1:
                self.belief_food.add((r, c))
            else:
                self.belief_food.discard((r, c))

        opp_cells = {
            (int(r), int(c)) for r, c in np.argwhere(obs[:, :, self.OPP_CH] == 1)
        }
        # Merge newly seen opponents, then forget stale cells we can now see
        # and found empty.
        merged = self.belief_opp | opp_cells
        self.belief_opp = {
            (r, c)
            for (r, c) in merged
            if not (vis[r, c] == 1 and (r, c) not in opp_cells)
        }
        if opp_cells:
            self.opp_age = 0
        else:
            self.opp_age += 1

    def channels(self):
        out = np.zeros((self.H, self.W, 3), dtype=np.float32)
        for r, c in self.belief_food:
            out[r, c, 0] = 1.0
        for r, c in self.belief_opp:
            out[r, c, 1] = 1.0
        out[:, :, 2] = min(self.opp_age, self.horizon) / self.horizon
        return out

    def augment(self, obs):
        return np.concatenate([obs.astype(np.float32), self.channels()], axis=-1)

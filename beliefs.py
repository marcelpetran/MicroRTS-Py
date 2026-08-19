import numpy as np


class BeliefTracker:
    """
    Belief channels derived ONLY from the agent's own observation history.
    Update rules mirror the heuristic opponents' beliefs:
      - food: add where food observed, discard where cell observed empty [6]
      - opponent: last-seen position + steps-since-seen [6]

    Channels (H, W, 3):
      0: belief_food     cells believed to contain food
      1: opp_last_seen   1.0 at opponent's last observed cell
      2: opp_age         steps since opponent last seen, normalized to [0, 1]
    """

    FOOD_CH, OPP_CH, VIS_CH = 1, 3, 5

    def __init__(self, height, width, map_layout=None, horizon=50):
        self.H, self.W = height, width
        self.horizon = horizon
        self.prior_food, self.prior_opp = set(), None
        if map_layout is not None:  # same 'o'/'B' static prior heuristics get [6]
            for r, row in enumerate(map_layout):
                for c, ch in enumerate(row):
                    if ch == "o":
                        self.prior_food.add((r, c))
                    elif ch == "B":  # learner is agent 0 -> opponent starts at 'B'
                        self.prior_opp = (r, c)
        self.reset()

    def reset(self, use_map_prior=True):
        self.belief_food = set(self.prior_food) if use_map_prior else set()
        self.opp_last_seen = self.prior_opp if use_map_prior else None
        self.opp_age = 0

    def update(self, obs):
        vis = obs[:, :, self.VIS_CH]
        for r, c in np.argwhere(vis == 1):
            if obs[r, c, self.FOOD_CH] == 1:
                self.belief_food.add((r, c))
            else:
                self.belief_food.discard((r, c))
        opp = np.argwhere(obs[:, :, self.OPP_CH] == 1)
        self.opp_age += 1
        if len(opp) > 0:
            self.opp_last_seen = tuple(opp[0])
            self.opp_age = 0

    def channels(self):
        out = np.zeros((self.H, self.W, 3), dtype=np.float32)
        for r, c in self.belief_food:
            out[r, c, 0] = 1.0
        if self.opp_last_seen is not None:
            r, c = self.opp_last_seen
            out[r, c, 1] = 1.0
        out[:, :, 2] = min(self.opp_age, self.horizon) / self.horizon
        return out

    def augment(self, obs):
        return np.concatenate([obs, self.channels()], axis=-1)

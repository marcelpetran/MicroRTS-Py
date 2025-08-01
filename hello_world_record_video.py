import numpy as np
import imageio
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

env = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=1,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.coacAI],
    # map_paths=["maps/16x16/basesWorkers16x16.xml"],
    map_paths=["maps/4x4/base4x4.xml"],
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
)

obs = env.reset()
frames = []

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def sample(logits):
    # https://stackoverflow.com/a/40475357/6611317
    p = softmax(logits, axis=1)
    c = p.cumsum(axis=1)
    u = np.random.rand(len(c), 1)
    choices = (u < c).argmax(axis=1)
    return choices.reshape(-1, 1)

# Step through and manually grab each frame:
for _ in range(2000):
    action_mask = env.get_action_mask()
    action_mask = action_mask.reshape(-1, action_mask.shape[-1])
    action_mask[action_mask == 0] = -9e8
    # sample valid actions
    action = np.concatenate(
        (
            sample(action_mask[:, 0:6]),  # action type
            sample(action_mask[:, 6:10]),  # move parameter
            sample(action_mask[:, 10:14]),  # harvest parameter
            sample(action_mask[:, 14:18]),  # return parameter
            sample(action_mask[:, 18:22]),  # produce_direction parameter
            sample(action_mask[:, 22:29]),  # produce_unit_type parameter
            # attack_target parameter
            sample(action_mask[:, 29 : sum(env.action_space.nvec[1:])]),
        ),
        axis=1,
    )
    obs, reward, done, info = env.step(action)
    # grab a numpy image instead of the default human render
    frame = env.render(mode='rgb_array')
    frames.append(frame)
    if done:
        break

# 3) Save as mp4 at 30 fps
imageio.mimsave('microrts_episode.mp4', frames, fps=30)

env.close()
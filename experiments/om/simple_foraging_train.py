from simple_foraging_env import SimpleForagingEnv
from opponent_model import OpponentModel, SubGoalSelector
from q_agent import QLearningAgent, OMGArgs, ReplayBuffer
import experiments.om.transformers as t  # your module name
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

env = SimpleForagingEnv(grid_size=5, max_steps=50)

obs_sample = env.reset()
H, W, F = obs_sample[0].shape
NUM_ACTIONS = 4 # Up, Down, Left, Right

args = OMGArgs(
    horizon_H=3,
    selector_mode="conservative",
    state_feature_splits=(F,),
    g_dim=16, # Latent dimension
    alpha=0.1, # Weight for the KL loss
    eta=1.0, # Starting value for eta (will anneal in agent)
)

# VAE (Teacher)
vae = t.TransformerVAE(
    h=H, w=W, feature_split_sizes=args.state_feature_splits,
    latent_dim=args.g_dim, d_model=64, nhead=4, num_encoder_layers=2,
    num_decoder_layers=2, dim_feedforward=128, dropout=0.1
).to(device)

# CVAE (Student)
cvae = t.TransformerCVAE(
    h=H, w=W, state_feature_splits=args.state_feature_splits,
    num_actions=NUM_ACTIONS, latent_dim=args.g_dim, d_model=64, nhead=4,
    num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=128, dropout=0.1
).to(device)

# --- Pre-train the VAE ---
print("Pre-training VAE...")
vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
vae_replay = ReplayBuffer(10000)

t.train_vae(env, vae, vae_replay, vae_optimizer, num_epochs=5000)
print("VAE pre-training complete.")

selector = SubGoalSelector(args)
cvae_optimizer = torch.optim.Adam(cvae.parameters(), lr=3e-4)

op_model = OpponentModel(cvae, vae, selector, optimizer=cvae_optimizer, device=device, args=args)
agent = QLearningAgent(env, op_model, device=device, args=args)

for ep in range(10000):
    stats = agent.run_episode(max_steps=50)
    if ep % 50 == 0:
        print(f"Episode {ep}: Return={stats['return']:.2f}, Steps={stats['steps']}")
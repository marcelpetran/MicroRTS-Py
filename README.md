# Opponent Modeling based on Subgoal Inference

This project explores opponent modeling in multi-agent reinforcement learning, specifically within the context of what we define as **Explicit Sequential Subgoal (ESS) Games**. In these games, an agent's behavior can be effectively modeled as a sequence of intermediate objectives or "subgoals." By inferring these subgoals, an agent can better predict an opponent's long-term strategy and adapt its own policy for more effective counter-play.

This repository provides the implementation of a Q-learning agent augmented with a Transformer-based opponent model, trained and evaluated in a competitive foraging environment.

For a detailed theoretical background on ESS games and the foundational concepts of this work, please refer to the paper located at `papers/Opponent_Modeling_in_Zero_Sum_Games.pdf`. Please note that while the core concepts remain, the specific architecture described in the paper has been updated in this implementation.

## Architecture

The system is composed of two main components: a `QLearningAgent` that learns to act in the environment, and an `OpponentModel` that provides insights into the opponent's intentions.

### Opponent Model

The `OpponentModel` is the core of our subgoal inference mechanism. It is designed to predict the opponent's next subgoal based on their recent trajectory.

-   **Model:** It uses a Transformer-based architecture, specifically a `SpatialOpponentModel` (`transformers.py`), to process a sequence of historical states and opponent actions.
-   **Input:** The model takes the current state and a history of the opponent's recent (state, action) pairs.
-   **Output:** It produces a spatial heatmap (`g_map`) over the game grid, representing a probability distribution over the opponent's potential subgoal locations.
-   **Training:** The model is trained using a cross-entropy loss, supervised via "hindsight." At the end of an episode, the opponent's actual success (i.e., which food they collected) is used as the ground-truth label for what their subgoal was.

### Q-Learning Agent

The `QLearningAgent` is a Double Deep Q-Network (DDQN) agent responsible for making decisions in the environment. It is augmented to leverage the predictions of the `OpponentModel`.

-   **Q-Network:** The agent's `QNet` is a Convolutional Neural Network (CNN) that approximates the action-value function `Q(s, g, a)`.
-   **Conditioned Input:** Crucially, the Q-network takes not only the current state `s` but also the subgoal heatmap `g` (produced by the `OpponentModel`) as input. This allows the agent's policy to be conditioned on the inferred intent of the opponent.
-   **Training:** The agent is trained using a replay buffer and DDQN to learn a policy that maximizes its own rewards, taking the opponent's predicted strategy into account.

## Environment

The experiments are conducted in the `SimpleForagingEnv`, a grid-world environment where two agents compete to collect food items.

-   **Objective:** Be the first to collect the food items.
-   **Setup:** A 2-player, zero-sum game on a grid. Each player starts at a fixed position, and two food items are placed at other fixed positions.
-   **Actions:** Agents can move Up, Down, Left, or Right.

## How to Run

The main training script is `simple_foraging_train.py`. You can configure the training run using command-line arguments.

**Example:**

```bash
python simple_foraging_train.py --episodes 20000 --batch_size 32 --d_model 256
```

### Key Arguments:

-   `--oracle`: Use a ground-truth "oracle" opponent model instead of the learned Transformer model.
-   `--classic`: Run a standard Q-learning agent without any opponent modeling. This is useful as a baseline.
-   `--episodes`: Number of episodes to train for.
-   `--env_size`: The size of the grid (e.g., 11 for an 11x11 grid).
-   `--batch_size`: The batch size for training the networks.
-   `--d_model`, `--nhead`, etc.: Hyperparameters for the Transformer architecture.

## Key Files

-   `simple_foraging_train.py`: The main script to start the training process.
-   `q_agent.py`: Contains the implementation of the `QLearningAgent`, including the `QNet` and replay buffer.
-   `opponent_model.py`: Implements the `OpponentModel`, which wraps the Transformer network and handles the training step.
-   `transformers.py`: Defines the `SpatialOpponentModel`, the Transformer-based network for subgoal inference.
-   `simple_foraging_env.py`: Defines the `SimpleForagingEnv` game environment.
-   `omg_args.py`: A dataclass for managing hyperparameters and configuration.
-   `papers/Opponent_Modeling_in_Zero_Sum_Games.pdf`: The research paper detailing the theoretical foundations of the project.

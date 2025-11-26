from typing import Optional

import numpy as np
import numpy.typing as npt
from gg_core import Action, GameState, GGConfig
from tqdm import tqdm

ACTIONS = [Action.Up, Action.Right, Action.Down, Action.Left]


def initialize_q_table(config: GGConfig) -> npt.NDArray[np.float32]:
    """
    Initialize and return an all-zero Q-table.

    Args:
        config: Game configuration; use config.world_generation.size -> (width, height).

    Returns:
        Q-table of zeros with dtype float32.
    """
    width, height = config.world_generation.size
    return np.zeros((width, height, width, height, len(ACTIONS)), dtype=np.float32)


def epsilon_greedy(
    rng: np.random.Generator, q_values: npt.NDArray[np.float32], epsilon: float
) -> int:
    """
    Select an action index using the epsilon-greedy strategy.

    Args:
        rng: Numpy random generator for reproducibility.
        q_values: 1D array with shape (|A|,) representing Q(s, ·).
        epsilon: Exploration rate in [0, 1].

    Returns:
        The selected action index (0..|A|-1).
    """
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    return int(np.argmax(q_values))


def select_intended_action(
    q: npt.NDArray[np.float32],
    state: GameState,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """
    Choose the intended action A_t via epsilon-greedy on Q(s, ·) at the agent's position.

    Args:
        q: Q-table of shape (width, height, |A|).
        state: Current GameState.
        epsilon: Exploration rate.
        rng: Random generator.

    Returns:
        Intended action index (0..|A|-1).
    """
    ax, ay = state.board.agent_position
    gx, gy = state.board.ghost_position or (0, 0)
    return epsilon_greedy(rng, q[ax, ay, gx, gy, :], epsilon)


def sample_actual_action_index(
    intended_idx: int, transition: list[float], rng: np.random.Generator
) -> int:
    """
    Sample the environment's executed action index given the intended one.

    Args:
        intended_idx: Behavior policy's chosen action index A_t.
        transition: Base transition probabilities for [intended, right, back, left].
        rng: Random generator.

    Returns:
        Actual executed action index (0..|A|-1).
    """
    a, b, c, d = transition
    slip_table = (
        [a, b, c, d],
        [d, a, b, c],
        [c, d, a, b],
        [b, c, d, a],
    )
    return int(rng.choice(4, p=slip_table[intended_idx]))


def compute_td_target(
    reward_next: float,
    next_state: GameState,
    q: npt.NDArray[np.float32],
    gamma: float,
) -> float:
    """
    Compute the Q-learning TD target in the form shown in the diagram:
      target = R_{t+1} + γ max_a Q(S_{t+1}, a)  (or just R_{t+1} if terminal)

    Args:
        reward_next: R_{t+1}, the reward when arriving at next_state.
        next_state: Successor state S_{t+1}.
        q: Q-table.
        gamma: Discount factor.

    Returns:
        Scalar TD target.
    """
    if next_state.done:
        return float(reward_next)
    nx, ny = next_state.board.agent_position
    gx, gy = next_state.board.ghost_position or (0, 0)
    return float(reward_next + gamma * np.max(q[nx, ny, gx, gy, :]))


def apply_q_update(
    q: npt.NDArray[np.float32],
    state: GameState,
    intended_idx: int,
    alpha: float,
    td_target: float,
) -> None:
    """
    Apply the tabular Q-learning update in place at (state, intended_idx):
      Q(S_t, A_t) ← (1 − α) Q(S_t, A_t) + α · target

    What to implement:
    - Locate (x, y) via state.board.agent_position and update the Q-table entry.

    Args:
        q: Q-table array.
        state: Current state S_t.
        intended_idx: Action index A_t.
        alpha: Learning rate α.
        td_target: The TD target computed for this transition.
    """
    x, y = state.board.agent_position
    gx, gy = state.board.ghost_position or (0, 0)
    qval = q[x, y, gx, gy, intended_idx]
    q[x, y, gx, gy, intended_idx] = (1 - alpha) * qval + alpha * td_target


def extract_policy_from_q(q: npt.NDArray[np.float32], config: GGConfig) -> np.ndarray:
    """
    Build a greedy policy π from Q by choosing argmax over actions per agent cell,
    averaging over ghost positions.

    What to implement:
    - For each (x, y), compute best_idx = argmax_a E_{gx,gy}[ Q[x, y, gx, gy, a] ]
      and map to Action.

    Args:
        q: Q-table of shape (width, height, width, height, |A|).
        config: Provides (width, height) via config.world_generation.size.

    Returns:
        A policy grid (width, height) of Action objects.
    """
    width, height = config.world_generation.size
    policy = np.empty((width, height), dtype=object)
    for x in range(width):
        for y in range(height):
            expected = np.mean(q[x, y, :, :, :], axis=(0, 1))
            best_idx = int(np.argmax(expected))
            policy[x, y] = ACTIONS[best_idx]
    return policy


def q_learning_step(
    q: npt.NDArray[np.float32],
    state: GameState,
    config: GGConfig,
    alpha: float,
    gamma: float,
    epsilon: float,
    rng: np.random.Generator,
) -> GameState:
    """
    Perform a single Q-learning transition and update.

    What to implement:
    - Select intended action via select_intended_action.
    - Sample actual executed action with sample_actual_action_index using
      config.agent.transition.
    - Step to next_state; get reward_next = next_state.reward.
    - Determine if next_state is terminal; compute TD target accordingly.
    - Update Q at (state, intended_idx); return next_state.

    Args:
        q: Q-table to update.
        state: Current state S_t.
        config: Configuration (transition model is in config.agent.transition).
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Exploration rate for epsilon-greedy.
        rng: Random generator.

    Returns:
        The next GameState S_{t+1}.
    """
    intended_idx = select_intended_action(q, state, epsilon, rng)
    actual_idx = sample_actual_action_index(intended_idx, list(config.agent.transition), rng)
    next_state = state.next_state(ACTIONS[actual_idx])
    td_target = compute_td_target(next_state.reward, next_state, q, gamma)
    apply_q_update(q, state, intended_idx, alpha, td_target)
    return next_state


def train(
    config: GGConfig,
    initial_state: GameState,
    episodes: int = 800000,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.2,
    max_steps_per_episode: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Orchestrate Q-learning over multiple episodes and return a greedy policy.

    What to implement:
    - Initialize Q with initialize_q_table.
    - For each episode, reset to initial_state and repeat q_learning_step up to
        max_steps_per_episode or until state.done is True.
    - After training, build π with extract_policy_from_q.

    Args:
        config: Game configuration (world size and transition probabilities).
        initial_state: Starting GameState for each episode.
        episodes: Number of training episodes to run.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Exploration rate (constant in this trainer).
        max_steps_per_episode: Optional cap on steps per episode (defaults to width*height).
        seed: Optional RNG seed; falls back to config.generation_seed if present.

    Returns:
        A 2D array policy with shape (width, height) populated by Action objects.
    """
    width, height = config.world_generation.size
    q = initialize_q_table(config)
    rng = np.random.default_rng(seed or getattr(config, "generation_seed", 0))
    max_steps = max_steps_per_episode or (width * height)

    for _ in tqdm(range(episodes), desc="Training Q-learning"):
        s, _ = initial_state.reset()

        for _ in range(max_steps):
            s_next = q_learning_step(q, s, config, alpha, gamma, epsilon, rng)

            if isinstance(s_next, tuple):
                s_next = s_next[0]

            s = s_next
            if s.done:
                break

    policy = extract_policy_from_q(q, config)

    return policy

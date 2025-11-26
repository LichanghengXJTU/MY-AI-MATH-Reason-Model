# root/src/gg/q_learning_linear.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import numpy.typing as npt
from gg_core import Action, GameState, GGConfig
from tqdm import tqdm

# Reuse the slip kernel and epsilon-greedy utilities from the tabular implementation.
from gg.q_learning import (
    epsilon_greedy,
    sample_actual_action_index,
    initialize_q_table,
    q_learning_step,
)

# Action ordering is identical to gg.q_learning so that the slip kernel stays consistent.
ACTIONS = [Action.Up, Action.Right, Action.Down, Action.Left]


@dataclass
class ToyTrainingConfig:
    """Configuration for the linear-function-approximation toy model.

    This configuration mirrors the notation in the paper:

      - alpha: learning rate η
      - gamma: discount factor γ
      - epsilon: exploration rate ε for the behaviour policy μ_t
      - feature_dim: dimensionality N of the Gaussian feature map φ(s, a)
      - episodes: number of training episodes
      - max_steps_per_episode: safety cap on episode length
      - eval_interval: how often (in episodes) to record loss diagnostics
    """

    alpha: float = 1e-3
    gamma: float = 0.9
    epsilon: float = 0.2
    feature_dim: int = 64
    episodes: int = 300_000
    max_steps_per_episode: Optional[int] = None
    eval_interval: int = 2_000
    seed: Optional[int] = None
    # Which hyperparameter sweep this config belongs to (for plotting).
    # E.g. 'alpha', 'epsilon', 'gamma', or 'feature_dim'.
    sweep: str = "alpha"



def initialize_gaussian_features(
    config: GGConfig,
    feature_dim: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float32]:
    """Sample a fixed Gaussian feature tensor φ(s, a).

    We use DMFT-style scaling so that each feature vector φ(s, a) has
    O(1) norm instead of growing like √N:
        φ_i(s, a) ~ N(0, 1 / sqrt(N)).
    This keeps the initial Q-values O(1) and greatly reduces numeric overflow
    when using large feature_dim.
    """
    width, height = config.world_generation.size

    # DMFT-style normalization: entries ~ N(0, 1/sqrt(N))
    scale = 1.0 / np.sqrt(feature_dim)
    phi = rng.normal(
        loc=0.0,
        scale=scale,
        size=(width, height, width, height, len(ACTIONS), feature_dim),
    ).astype(np.float32)
    return phi



def q_value_linear(
    w: npt.NDArray[np.float32],
    phi_sa: npt.NDArray[np.float32],
) -> float:
    """Compute Q_w(s, a) = w^T φ(s, a) for a single feature vector."""
    return float(np.dot(w, phi_sa))


def q_values_linear_for_state(
    w: npt.NDArray[np.float32],
    phi: npt.NDArray[np.float32],
    state: GameState,
) -> npt.NDArray[np.float32]:
    """Return the vector Q_w(S_t, ·) for the current agent/ghost positions.

    This matches the role of q[ax, ay, gx, gy, :] in the tabular implementation.
    """
    ax, ay = state.board.agent_position
    gx, gy = state.board.ghost_position or (0, 0)
    # Shape: (|A|, N)
    feats = phi[ax, ay, gx, gy, :, :]
    # (|A|, N) @ (N,) -> (|A|,)
    return feats @ w


def linear_td_target(
    w: npt.NDArray[np.float32],
    phi: npt.NDArray[np.float32],
    reward_next: float,
    next_state: GameState,
    gamma: float,
) -> float:
    """Compute the Q-learning TD target for the linear model.

    This implements the target G_t(w) from Sec.~2.2:

      G_t(w) = R_{t+1} + γ max_{a'} Q_w(S_{t+1}, a'),

    with the convention that if next_state.done is True, we drop the bootstrap term
    and return just R_{t+1}.
    """
    if next_state.done:
        return float(reward_next)

    ax, ay = next_state.board.agent_position
    gx, gy = next_state.board.ghost_position or (0, 0)
    feats_next = phi[ax, ay, gx, gy, :, :]  # (|A|, N)
    q_next = feats_next @ w  # (|A|,)
    return float(reward_next + gamma * float(np.max(q_next)))


def q_learning_step_linear(
    w: npt.NDArray[np.float32],
    phi: npt.NDArray[np.float32],
    state: GameState,
    config: GGConfig,
    alpha: float,
    gamma: float,
    epsilon: float,
    rng: np.random.Generator,
    behaviour: str = "unfixed",
    q_teacher: Optional[npt.NDArray[np.float32]] = None,
) -> tuple[GameState, float, float]:
    """Single semi-gradient Q-learning step under linear function approximation.

    This mirrors the derivation in Sec.~2.2 (TD Update and Semi-Gradient Derivation):

      δ_t^{QL}(w) = R_{t+1} + γ max_a Q_w(S_{t+1}, a) − Q_w(S_t, A_t),
      ∇_w J_t(w) ≈ −δ_t^{QL}(w) φ(S_t, A_t),

    and the SGD update

      w_{t+1} = w_t + α δ_t^{QL}(w_t) φ(S_t, A_t).

    We keep the environment transition model identical to the tabular q_learning_step
    by reusing the same slip kernel and Action ordering.
    """
    # 1. Behaviour policy μ_t: epsilon-greedy in Q_w(S_t, ·)
    if behaviour == "fixed" and q_teacher is not None:
        ax, ay = state.board.agent_position
        gx, gy = state.board.ghost_position or (0, 0)
        q_vals_behaviour = q_teacher[ax, ay, gx, gy, :]  # (|A|,)
        intended_idx = epsilon_greedy(
            rng, q_vals_behaviour.astype(np.float32), epsilon
        )
    else:
        q_values = q_values_linear_for_state(w, phi, state)
        intended_idx = epsilon_greedy(
            rng, q_values.astype(np.float32), epsilon
        )

    # 2. Sample actual executed action using the slip kernel.
    actual_idx = sample_actual_action_index(
        intended_idx, list(config.agent.transition), rng
    )
    action = ACTIONS[actual_idx]

    # 3. Step the environment.
    next_state = state.next_state(action)
    reward_next = float(next_state.reward)

    # 4. Compute TD error δ_t^{QL}(w).
    td_target = linear_td_target(w, phi, reward_next, next_state, gamma)
    ax, ay = state.board.agent_position
    gx, gy = state.board.ghost_position or (0, 0)
    phi_sa = phi[ax, ay, gx, gy, intended_idx, :]  # (N,)
    q_sa = float(np.dot(w, phi_sa))
    delta = td_target - q_sa

    # 5. Semi-gradient update: w ← w + α δ φ(S_t, A_t).
    w += alpha * delta * phi_sa

    return next_state, float(delta), reward_next


def _build_reference_indices(
    config: GGConfig,
    stride: int = 1,
) -> npt.NDArray[np.int32]:
    """Construct a reference set of (ax, ay, gx, gy, a_idx) indices.

    These indices define the fixed reference distribution d_ref over state–action
    pairs used in the prediction-error loss:

      L_pred(w) = E_{(s,a) ~ d_ref}[(Q_teacher(s,a) − Q_w(s,a))^2].

    For simplicity we build a regular grid over all positions with a configurable
    stride. Unreachable states are harmless: they simply appear in d_ref with
    non-zero weight but do not affect the training dynamics.
    """
    width, height = config.world_generation.size
    indices = []
    for ax in range(0, width, stride):
        for ay in range(0, height, stride):
            for gx in range(0, width, stride):
                for gy in range(0, height, stride):
                    for a_idx in range(len(ACTIONS)):
                        indices.append((ax, ay, gx, gy, a_idx))
    return np.asarray(indices, dtype=np.int32)


def _q_linear_for_indices(
    w: npt.NDArray[np.float32],
    phi: npt.NDArray[np.float32],
    indices: npt.NDArray[np.int32],
) -> npt.NDArray[np.float32]:
    """Evaluate Q_w(s, a) on a batch of indices.

    Args:
        w: Parameter vector of shape (N,).
        phi: Feature tensor of shape (W, H, W, H, |A|, N).
        indices: Integer array of shape (K, 5) with columns (ax, ay, gx, gy, a_idx).

    Returns:
        1D float32 array of shape (K,) with Q_w(s, a) values.
    """
    ax = indices[:, 0]
    ay = indices[:, 1]
    gx = indices[:, 2]
    gy = indices[:, 3]
    a_idx = indices[:, 4]
    feats = phi[ax, ay, gx, gy, a_idx, :]  # (K, N)
    return (feats @ w).astype(np.float32)


def train_tabular_teacher(
    config: GGConfig,
    initial_state: GameState,
    episodes: int = 200_000,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    max_steps_per_episode: Optional[int] = None,
    seed: Optional[int] = None,
) -> npt.NDArray[np.float32]:
    """Train a tabular Q-learning teacher Q_table.

    This reuses the low-level tabular primitives from gg.q_learning so that the
    teacher matches exactly the behaviour of the baseline Q-learning agent, but
    with more conservative hyperparameters (smaller α, ε) and many more episodes
    to approximate convergence to Q*.

    The resulting Q-table has shape (W, H, W, H, |A|) and dtype float32.
    """
    if max_steps_per_episode is None:
        max_steps = getattr(config.agent, "max_steps_per_episode", None) or 1_000
    else:
        max_steps = max_steps_per_episode

    rng = np.random.default_rng(seed)
    q = initialize_q_table(config)

    for _ in tqdm(range(episodes), desc="Training tabular teacher Q"):
        state, _ = initial_state.reset()

        for _ in range(max_steps):
            next_state = q_learning_step(q, state, config, alpha, gamma, epsilon, rng)

            # The Rust binding sometimes returns (state, debug_info); be robust.
            if isinstance(next_state, tuple):
                next_state = next_state[0]

            state = next_state
            if state.done:
                break

    return q.astype(np.float32)


def train_linear_toy(
    config: GGConfig,
    initial_state: GameState,
    q_teacher: npt.NDArray[np.float32],
    output_dir: Path,
    toy_cfg: Optional[ToyTrainingConfig] = None,
    policy_mode: str = "unfixed",
) -> Dict[str, Any]:
    """Train the linear-function-approximation toy model and log diagnostics.

    This function implements the unfixed-policy TD(0) control toy model described
    in Sec.~2 and Sec.~\\ref{sec:toy-losses}:

      - Features φ(s, a) are random Gaussian vectors shared across the entire run.
      - The behaviour policy μ_t is ε-greedy in Q_w(S_t, ·).
      - The target policy π_t is greedy in Q_w (implicitly through the TD target).
      - Both μ_t and π_t are recomputed from Q_w at every step, realizing the
        second type of non-stationarity ("unfixed policy").

    We track the following quantities at evaluation checkpoints n:

      - L_TD(n): empirical mean of (δ_t^{QL}(w))^2 between checkpoints.
      - L_pred(n): prediction-error loss vs the tabular teacher Q_teacher.
      - ||w_n||^2: squared parameter norm.
      - Q_max(n): max |Q_w(s,a)| over the reference set.
      - Return(n): mean episodic return over episodes in the last interval.

    All diagnostics are saved to output_dir / "toy_linear" / "logs.npz" along with
    a meta.json containing hyperparameters.
    """
    if toy_cfg is None:
        toy_cfg = ToyTrainingConfig()

    output_dir = Path(output_dir)
    toy_dir = output_dir / "toy_linear"
    toy_dir.mkdir(parents=True, exist_ok=True)

    # Random generator and feature map.
    rng = np.random.default_rng(toy_cfg.seed)
    phi = initialize_gaussian_features(config, toy_cfg.feature_dim, rng)

    # Initialize w with the same 1/sqrt(N) scaling to keep initial Q_w(s,a)
    # on the order of 1 instead of √N. This matches the linear-theory/DMFT
    # conventions and avoids huge initial TD targets.
    w = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(toy_cfg.feature_dim),
        size=(toy_cfg.feature_dim,),
    ).astype(np.float32)

    # Reference indices and teacher values for L_pred.
    ref_indices = _build_reference_indices(config, stride=1)
    q_teacher_ref = q_teacher[
        ref_indices[:, 0],
        ref_indices[:, 1],
        ref_indices[:, 2],
        ref_indices[:, 3],
        ref_indices[:, 4],
    ].astype(np.float32)

    if toy_cfg.max_steps_per_episode is None:
        max_steps = getattr(config.agent, "max_steps_per_episode", None) or 1_000
    else:
        max_steps = toy_cfg.max_steps_per_episode

    # Logging buffers.
    checkpoints: list[int] = []
    td_losses: list[float] = []
    pred_losses: list[float] = []
    w_norm2_list: list[float] = []
    q_abs_max_list: list[float] = []
    avg_returns: list[float] = []

    # Running accumulators between checkpoints.
    running_delta2 = 0.0
    running_count = 0
    recent_returns: list[float] = []

    for episode in tqdm(range(toy_cfg.episodes), desc="Training linear TD control"):
        state, _ = initial_state.reset()
        episode_return = 0.0

        for _ in range(max_steps):
            next_state, delta, reward = q_learning_step_linear(
                w,
                phi,
                state,
                config,
                toy_cfg.alpha,
                toy_cfg.gamma,
                toy_cfg.epsilon,
                rng,
                behaviour=policy_mode,
                q_teacher=q_teacher,
            )

            if isinstance(next_state, tuple):
                next_state = next_state[0]

            state = next_state
            episode_return += float(reward)

            running_delta2 += float(delta * delta)
            running_count += 1

            if state.done:
                break

        recent_returns.append(episode_return)

        if (episode + 1) % toy_cfg.eval_interval == 0:
            # Empirical TD loss based on deltas seen since the last checkpoint.
            mean_delta2 = running_delta2 / max(running_count, 1)
            L_td = 0.5 * mean_delta2

            running_delta2 = 0.0
            running_count = 0

            # Prediction-error loss vs teacher.
            q_student_ref = _q_linear_for_indices(w, phi, ref_indices)
            L_pred = float(np.mean((q_teacher_ref - q_student_ref) ** 2))

            # Parameter norm and max |Q|.
            w_norm2 = float(np.dot(w, w))
            q_abs_max = float(np.max(np.abs(q_student_ref)))

            # Average episodic return over the last interval.
            avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
            recent_returns = []

            checkpoints.append(episode + 1)
            td_losses.append(L_td)
            pred_losses.append(L_pred)
            w_norm2_list.append(w_norm2)
            q_abs_max_list.append(q_abs_max)
            avg_returns.append(avg_return)

    # Convert to arrays and save.
    logs_path = toy_dir / "logs.npz"
    np.savez(
        logs_path,
        episodes=np.asarray(checkpoints, dtype=np.int64),
        td_loss=np.asarray(td_losses, dtype=np.float32),
        pred_loss=np.asarray(pred_losses, dtype=np.float32),
        w_norm2=np.asarray(w_norm2_list, dtype=np.float32),
        q_abs_max=np.asarray(q_abs_max_list, dtype=np.float32),
        avg_return=np.asarray(avg_returns, dtype=np.float32),
    )

    # Save model parameters (w and φ) for possible post-hoc analysis (we only keep φ.shape).
    model_path = toy_dir / "model.npz"
    np.savez(
        model_path,
        w=w,
        phi_shape=np.asarray(phi.shape, dtype=np.int64),
    )

    meta: Dict[str, Any] = {
        "alpha": toy_cfg.alpha,
        "gamma": toy_cfg.gamma,
        "epsilon": toy_cfg.epsilon,
        "feature_dim": toy_cfg.feature_dim,
        "episodes": toy_cfg.episodes,
        "eval_interval": toy_cfg.eval_interval,
        "max_steps_per_episode": toy_cfg.max_steps_per_episode,
        "seed": toy_cfg.seed,
        "ref_indices_count": int(ref_indices.shape[0]),
        "sweep": getattr(toy_cfg, "sweep", "unspecified"),
        "policy_mode": policy_mode,
    }
    meta_path = toy_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        import json as _json

        _json.dump(meta, f, indent=2)

    return {
        "logs_path": logs_path,
        "model_path": model_path,
        "meta_path": meta_path,
        "meta": meta,
    }


def train_teacher_and_toy(
    config: GGConfig,
    initial_state: GameState,
    output_dir: Path,
    teacher_episodes: int = 200_000,
    teacher_alpha: float = 0.1,
    teacher_gamma: float = 0.9,
    teacher_epsilon: float = 0.1,
    teacher_max_steps_per_episode: Optional[int] = None,
    toy_cfg: Optional[ToyTrainingConfig] = None,
    seed: Optional[int] = None,
) -> None:
    """Convenience entry point used from __main__.py.

    This function

      1. Trains a conservative tabular Q-learning teacher Q_teacher.
      2. Saves Q_teacher and its metadata under output_dir / "teacher".
      3. Trains the linear toy model against this teacher and logs diagnostics.

    Both training procedures share the same initial_state / GGConfig and thus
    the same underlying MDP as in the main paper.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tabular teacher training.
    teacher_q = train_tabular_teacher(
        config=config,
        initial_state=initial_state,
        episodes=teacher_episodes,
        alpha=teacher_alpha,
        gamma=teacher_gamma,
        epsilon=teacher_epsilon,
        max_steps_per_episode=teacher_max_steps_per_episode,
        seed=seed,
    )

    teacher_dir = output_dir / "teacher"
    teacher_dir.mkdir(parents=True, exist_ok=True)

    teacher_q_path = teacher_dir / "q_teacher.npz"
    np.savez(teacher_q_path, q=teacher_q)

    teacher_meta = {
        "episodes": teacher_episodes,
        "alpha": teacher_alpha,
        "gamma": teacher_gamma,
        "epsilon": teacher_epsilon,
        "max_steps_per_episode": teacher_max_steps_per_episode,
        "seed": seed,
    }
    teacher_meta_path = teacher_dir / "meta.json"
    with teacher_meta_path.open("w", encoding="utf-8") as f:
        import json as _json

        _json.dump(teacher_meta, f, indent=2)

    # Linear toy training.
    train_linear_toy(
        config=config,
        initial_state=initial_state,
        q_teacher=teacher_q,
        output_dir=output_dir,
        toy_cfg=toy_cfg,
    )


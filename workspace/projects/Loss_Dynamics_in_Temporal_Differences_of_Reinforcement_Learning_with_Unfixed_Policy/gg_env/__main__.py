# root/src/gg/__main__.py

import os
from argparse import ArgumentParser
from importlib.metadata import version
from pathlib import Path
from typing import cast

import numpy as np
from gg_core import run, parse_config, GameState

from gg.q_learning import train as train_tabular_policy
from gg.q_learning_linear import train_teacher_and_toy, ToyTrainingConfig


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config file (e.g., configs/default.yaml).",
    )
    parser.add_argument(
        "--render-policy",
        type=str,
        help="Render an existing greedy policy stored in a .npz file.",
    )
    parser.add_argument(
        "--train-policy",
        type=str,
        help="Train a tabular Q-learning policy and save it to the given path.",
    )
    parser.add_argument(
        "--train-toy",
        type=str,
        help=(
            "Train both a tabular Q-learning teacher and the linear TD toy model. "
            "The argument is an output directory; it will be created if needed."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print gg package version and exit.",
    )

    args = parser.parse_args()

    if args.version:
        print(f"gg version: {version('gg')}")
        return

    config_path = Path(args.config)
    config = parse_config(str(config_path))

    if args.render_policy:
        # Load and render a pre-trained greedy policy.
        policy_path = Path(args.render_policy)
        with np.load(policy_path.as_posix()) as data:
            policy = data["policy"]
            seed = int(data["seed"][0])
        print(f"Rendering policy from {policy_path.as_posix()} (seed={seed}).")
        run(config, policy=policy)

    elif args.train_policy:
        # Train the baseline tabular Q-learning policy (original assignment).
        train_policy_path = Path(args.train_policy).with_suffix(".npz")
        config.headless = True
        initial_state, generation_seed, _episode_seed = cast(
            tuple[GameState, int, int],
            run(config),
        )
        print(
            f"Initial state generated with seed: {generation_seed}. "
            f"Will save trained policy to {train_policy_path.as_posix()}"
        )

        policy = train_tabular_policy(config, initial_state)
        os.makedirs(train_policy_path.parent, exist_ok=True)
        np.savez(train_policy_path.as_posix(), policy=policy, seed=[generation_seed])

    elif args.train_toy:
        # Jointly train teacher Q-table and linear toy model with diagnostics.
        output_dir = Path(args.train_toy)
        output_dir.mkdir(parents=True, exist_ok=True)

        config.headless = True
        initial_state, generation_seed, _episode_seed = cast(
            tuple[GameState, int, int],
            run(config),
        )
        print(
            f"Initial state generated with seed: {generation_seed}. "
            f"Training teacher Q and linear TD toy model into {output_dir.as_posix()}."
        )

        # You can change ToyTrainingConfig here to sweep different hyperparameters.
        toy_cfg = ToyTrainingConfig(
            alpha=1e-3,
            gamma=0.9,
            epsilon=0.2,
            feature_dim=64,
            episodes=300_000,
            max_steps_per_episode=None,
            eval_interval=2_000,
            seed=generation_seed,
        )
        train_teacher_and_toy(
            config=config,
            initial_state=initial_state,
            output_dir=output_dir,
            teacher_episodes=200_000,
            teacher_alpha=0.1,
            teacher_gamma=0.9,
            teacher_epsilon=0.1,
            teacher_max_steps_per_episode=None,
            toy_cfg=toy_cfg,
            seed=generation_seed,
        )

    else:
        raise RuntimeError(
            "One of --render-policy, --train-policy or --train-toy must be specified."
        )


if __name__ == "__main__":
    main()


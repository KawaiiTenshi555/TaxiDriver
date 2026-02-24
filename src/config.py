"""Shared project configuration constants."""

from __future__ import annotations

from typing import Final


TEST_SEED: Final[int] = 42
N_TEST_EPISODES: Final[int] = 100

OPTIMIZED_PARAMS: Final[dict[str, dict[str, float | int]]] = {
    "ql": {
        "alpha": 0.8,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.999,
        "epsilon_min": 0.01,
    },
    "dqn": {
        "lr": 0.001,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "batch_size": 64,
        "target_update_freq": 100,
    },
}

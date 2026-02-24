"""Shared utilities for rendering and replaying episodes in console."""

from __future__ import annotations

from typing import Callable

from environment.taxi_wrapper import TaxiWrapper


ACTION_NAMES = ["South", "North", "East", "West", "Pickup", "Dropoff"]


def run_console_replay(
    env: TaxiWrapper,
    policy: Callable[[int], int],
    title: str,
    seed: int | None = None,
) -> None:
    """Replay one episode in a dedicated ANSI environment."""
    render_env = TaxiWrapper(reward_mode=env.reward_mode, render_mode="ansi")
    state, _ = render_env.reset(seed=seed)
    total_reward = 0.0
    step = 0
    terminated = False
    truncated = False

    print("\n" + "-" * 40)
    print(f"  REPLAY - {title}")
    print("-" * 40)

    while not (terminated or truncated):
        print(render_env.render())
        action = policy(state)
        state, reward, terminated, truncated, _ = render_env.step(action)
        total_reward += reward
        step += 1
        print(
            f"  Step {step:3d} | Action: {ACTION_NAMES[action]:<8} | "
            f"Reward: {reward:+.0f} | Total: {total_reward:+.0f}"
        )

    print(render_env.render())
    result = "SUCCESS" if terminated else "FAILURE (timeout)"
    print(f"\n  Result: {result} in {step} steps | Total reward: {total_reward:+.0f}")
    print("-" * 40)
    render_env.close()

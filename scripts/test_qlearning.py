# test_qlearning.py — Test du Q-Learning (non-optimisé puis optimisé)

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from environment.taxi_wrapper import TaxiWrapper
from agents.q_learning import QLearningAgent

# ------------------------------------------------------------------
# Run 1 — Q-Learning NON-OPTIMISÉ (paramètres par défaut)
# ------------------------------------------------------------------
print("\n" + "=" * 50)
print("  RUN 1 — Q-Learning (paramètres par défaut)")
print("=" * 50)

env = TaxiWrapper(reward_mode="default")
agent = QLearningAgent(
    env,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    seed=42,
)
print(agent)

train_tracker = agent.train(n_episodes=1000, log_interval=100, seed=None)
test_tracker  = agent.test(n_episodes=100, log_interval=50, seed=42)

agent.save("results/models/qlearning_default.pkl")

# ------------------------------------------------------------------
# Run 2 — Q-Learning OPTIMISÉ (plus d'épisodes, meilleurs params)
# ------------------------------------------------------------------
print("\n" + "=" * 50)
print("  RUN 2 — Q-Learning (paramètres optimisés)")
print("=" * 50)

env2 = TaxiWrapper(reward_mode="default")
agent2 = QLearningAgent(
    env2,
    alpha=0.8,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.999,
    epsilon_min=0.01,
    seed=42,
)
print(agent2)

train_tracker2 = agent2.train(n_episodes=10000, log_interval=1000, seed=None)
test_tracker2  = agent2.test(n_episodes=100, log_interval=50, seed=42)

agent2.save("results/models/qlearning_optimized.pkl")

# ------------------------------------------------------------------
# Replay d'un épisode avec l'agent optimisé
# ------------------------------------------------------------------
print("\nReplay d'un épisode avec l'agent optimisé :")
agent2.play_episode(seed=7)

env.close()
env2.close()

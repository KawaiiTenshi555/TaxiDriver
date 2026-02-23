# test_bruteforce.py — Script de test rapide pour valider le setup

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from environment.taxi_wrapper import TaxiWrapper
from agents.brute_force import BruteForceAgent

# 1. Instanciation de l'environnement
env = TaxiWrapper(reward_mode="default")
print(f"Environnement : {env}")
print(f"  n_states  = {env.n_states}")
print(f"  n_actions = {env.n_actions}")

# 2. Test du décodage d'état
state, _ = env.reset(seed=42)
decoded = env.decode_state(state)
print(f"\nÉtat initial : {state}")
print(f"  Décodé     : {decoded}")

# 3. Test de l'agent Brute Force sur 200 épisodes
agent = BruteForceAgent(env, seed=42)
tracker = agent.test(n_episodes=200, log_interval=50, seed=42)

# 4. Replay d'un épisode aléatoire
print("\nReplay d'un épisode aléatoire :")
agent.play_episode(seed=7)

env.close()

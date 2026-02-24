# main.py — Point d'entrée du programme (mode User / mode Time-Limited)

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from environment.taxi_wrapper import TaxiWrapper
from agents.brute_force import BruteForceAgent
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from config import OPTIMIZED_PARAMS
from utils.metrics import MetricsTracker
from utils.visualization import (
    plot_learning_curves,
    plot_steps_evolution,
    plot_epsilon_decay,
    plot_summary_table,
    plot_steps_distribution,
)

# ==================================================================
# Helpers d'entrée utilisateur
# ==================================================================

def _ask_int(prompt: str, default: int, min_val: int = 1) -> int:
    while True:
        raw = input(f"{prompt} [défaut: {default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if val < min_val:
                print(f"  Valeur minimum : {min_val}")
                continue
            return val
        except ValueError:
            print("  Entier attendu.")


def _ask_float(prompt: str, default: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    while True:
        raw = input(f"{prompt} [défaut: {default}]: ").strip()
        if raw == "":
            return default
        try:
            val = float(raw)
            if not (min_val <= val <= max_val):
                print(f"  Valeur attendue entre {min_val} et {max_val}")
                continue
            return val
        except ValueError:
            print("  Nombre flottant attendu.")


def _ask_choice(prompt: str, choices: list[str]) -> int:
    for i, c in enumerate(choices, 1):
        print(f"  [{i}] {c}")
    while True:
        raw = input(f"{prompt}: ").strip()
        try:
            idx = int(raw)
            if 1 <= idx <= len(choices):
                return idx
        except ValueError:
            pass
        print(f"  Choix valide : 1 à {len(choices)}")


# ==================================================================
# Affichage de N replays aléatoires
# ==================================================================

def _show_replays(agent, n: int = 3) -> None:
    import random
    print(f"\n{'='*50}")
    print(f"  REPLAYS — {n} épisodes aléatoires")
    print(f"{'='*50}")
    for i in range(n):
        seed = random.randint(0, 9999)
        print(f"\n  --- Épisode {i+1}/{n} (seed={seed}) ---")
        agent.play_episode(seed=seed)


# ==================================================================
# Génération des graphiques après une session
# ==================================================================

def _generate_plots(train_tracker, test_tracker, agent, label: str) -> None:
    print("\nGénération des graphiques...")

    plot_learning_curves({label: train_tracker})
    plot_steps_evolution({label: train_tracker})

    if hasattr(agent, "epsilon_history") and agent.epsilon_history:
        plot_epsilon_decay({label: agent.epsilon_history})

    plot_steps_distribution({label: test_tracker})
    plot_summary_table({label: test_tracker.summary()})


# ==================================================================
# Mode 1 — User Mode
# ==================================================================

def run_user_mode() -> None:
    """
    Mode interactif : l'utilisateur configure tous les paramètres
    de l'algorithme, le nombre d'épisodes et le reward shaping.
    """
    print("\n" + "="*50)
    print("  MODE USER — Configuration manuelle")
    print("="*50)

    # Choix de l'algorithme
    print("\nAlgorithme :")
    algo_idx = _ask_choice("Choix", ["Brute Force", "Q-Learning", "DQN"])

    # Reward shaping
    print("\nReward shaping :")
    reward_idx = _ask_choice("Mode", ["default", "distance", "custom"])
    reward_modes = ["default", "distance", "custom"]
    reward_mode = reward_modes[reward_idx - 1]

    custom_rewards = None
    if reward_mode == "custom":
        print("\nDéfinir les récompenses custom :")
        custom_rewards = {
            "step":            _ask_float("  Pénalité par step",  default=-1.0, min_val=-100, max_val=0),
            "dropoff_success": _ask_float("  Bonus dropoff réussi", default=20.0, min_val=0,  max_val=200),
            "illegal_action":  _ask_float("  Pénalité action illégale", default=-10.0, min_val=-100, max_val=0),
        }

    # Épisodes
    print()
    n_train = _ask_int("Épisodes d'entraînement", default=5000)
    n_test  = _ask_int("Épisodes de test",         default=100)

    # Instanciation environnement
    env = TaxiWrapper(reward_mode=reward_mode, custom_rewards=custom_rewards)

    # ------ Brute Force ------
    if algo_idx == 1:
        agent = BruteForceAgent(env)
        print(f"\n[BruteForce] Lancement de {n_test} épisodes...")
        test_tracker = agent.test(n_episodes=n_test, log_interval=max(1, n_test // 5))
        _show_replays(agent)
        plot_steps_distribution({"BruteForce": test_tracker})
        plot_summary_table({"BruteForce": test_tracker.summary()})
        env.close()
        return

    # ------ Q-Learning ------
    if algo_idx == 2:
        print("\nHyperparamètres Q-Learning :")
        alpha         = _ask_float("  Alpha (learning rate)", default=0.8,   min_val=0.001, max_val=1.0)
        gamma         = _ask_float("  Gamma (discount)",      default=0.99,  min_val=0.0,   max_val=1.0)
        epsilon       = _ask_float("  Epsilon initial",       default=1.0,   min_val=0.0,   max_val=1.0)
        epsilon_decay = _ask_float("  Epsilon decay",         default=0.999, min_val=0.9,   max_val=1.0)
        epsilon_min   = _ask_float("  Epsilon min",           default=0.01,  min_val=0.0,   max_val=0.5)

        agent = QLearningAgent(
            env, alpha=alpha, gamma=gamma,
            epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
        )
        label = "Q-Learning (user)"

    # ------ DQN ------
    else:
        print("\nHyperparamètres DQN :")
        lr                 = _ask_float("  Learning rate",          default=0.001, min_val=1e-5, max_val=0.1)
        gamma              = _ask_float("  Gamma (discount)",        default=0.99,  min_val=0.0,  max_val=1.0)
        epsilon            = _ask_float("  Epsilon initial",         default=1.0,   min_val=0.0,  max_val=1.0)
        epsilon_decay      = _ask_float("  Epsilon decay",           default=0.995, min_val=0.9,  max_val=1.0)
        epsilon_min        = _ask_float("  Epsilon min",             default=0.01,  min_val=0.0,  max_val=0.5)
        batch_size         = _ask_int(  "  Batch size",              default=64,    min_val=16)
        target_update_freq = _ask_int(  "  Target update freq (steps)", default=100, min_val=10)

        agent = DQNAgent(
            env, lr=lr, gamma=gamma,
            epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
            batch_size=batch_size, target_update_freq=target_update_freq,
        )
        label = "DQN (user)"

    # Entraînement
    print(f"\n{agent}")
    train_tracker = agent.train(n_episodes=n_train, log_interval=max(1, n_train // 10))

    # Test
    test_tracker = agent.test(n_episodes=n_test, log_interval=max(1, n_test // 5))

    # Replays
    _show_replays(agent)

    # Graphiques
    _generate_plots(train_tracker, test_tracker, agent, label)

    env.close()


# ==================================================================
# Mode 2 — Time-Limited Mode
# ==================================================================


def run_time_limited_mode() -> None:
    """
    Mode time-limited : paramètres optimaux pré-fixés.
    L'utilisateur spécifie une durée maximale d'entraînement.
    Le programme s'entraîne jusqu'à expiration du temps, puis évalue.
    """
    print("\n" + "="*50)
    print("  MODE TIME-LIMITED — Paramètres optimisés")
    print("="*50)

    # Choix de l'algorithme (pas de brute force ici)
    print("\nAlgorithme :")
    algo_idx = _ask_choice("Choix", ["Q-Learning (optimisé)", "DQN (optimisé)"])

    time_budget = _ask_int("Durée d'entraînement (secondes)", default=60, min_val=5)
    n_test      = _ask_int("Épisodes de test",                 default=100)

    env = TaxiWrapper(reward_mode="default")

    if algo_idx == 1:
        agent = QLearningAgent(env, **OPTIMIZED_PARAMS["ql"])
        label = "Q-Learning (tuned)"
    else:
        agent = DQNAgent(env, **OPTIMIZED_PARAMS["dqn"])
        label = "DQN (tuned)"

    print(f"\n{agent}")
    print(f"Entraînement pendant {time_budget}s avec paramètres optimisés...")

    # Boucle d'entraînement bornée dans le temps
    tracker = MetricsTracker(n_episodes=999_999, log_interval=200, phase="Training")
    ep = 0
    deadline = time.perf_counter() + time_budget

    while time.perf_counter() < deadline:
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        tracker.begin_episode(ep)

        while not (terminated or truncated):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)

            if isinstance(agent, QLearningAgent):
                agent.update(state, action, reward, next_state, terminated or truncated)
            else:
                agent.buffer.push(state, action, reward, next_state, terminated or truncated)
                loss = agent.learn()
                if loss is not None:
                    agent.loss_history.append(loss)
                agent._step_count += 1
                if agent._step_count % agent.target_update_freq == 0:
                    agent._sync_target()

            state = next_state
            total_reward += reward
            steps += 1

        agent._decay_epsilon()
        tracker.end_episode(ep, steps, total_reward, terminated)
        ep += 1

    elapsed = time_budget - (deadline - time.perf_counter())
    print(f"\nEntraînement terminé — {ep} épisodes en {elapsed:.1f}s")

    # Test
    test_tracker = agent.test(n_episodes=n_test, log_interval=max(1, n_test // 5))

    # Replays
    _show_replays(agent)

    # Graphiques
    _generate_plots(tracker, test_tracker, agent, label)

    env.close()


# ==================================================================
# Point d'entrée
# ==================================================================

def main() -> None:
    print("\n" + "="*50)
    print("   TAXI DRIVER — Reinforcement Learning")
    print("   T-AIA-902")
    print("="*50)

    print("\nMode :")
    mode_idx = _ask_choice("Choix", ["User Mode (paramètres manuels)",
                                      "Time-Limited Mode (paramètres optimisés)"])

    if mode_idx == 1:
        run_user_mode()
    else:
        run_time_limited_mode()

    print("\nFin du programme.")


if __name__ == "__main__":
    main()

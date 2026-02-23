# benchmark.py — Benchmark complet : tous les algorithmes, toutes les métriques

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from environment.taxi_wrapper import TaxiWrapper
from agents.brute_force import BruteForceAgent
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from utils.visualization import (
    plot_learning_curves,
    plot_steps_evolution,
    plot_epsilon_decay,
    plot_summary_table,
    plot_steps_distribution,
    plot_hyperparam_heatmap,
    plot_reward_shaping_comparison,
)

# Seed commune pour tous les runs de test → comparaison équitable
TEST_SEED    = 42
N_TEST       = 100
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "results", "models")

# ==================================================================
# Stockage des résultats
# ==================================================================

train_trackers   = {}   # {label: MetricsTracker}  — courbes d'apprentissage
test_trackers    = {}   # {label: MetricsTracker}  — résultats de test
epsilon_histories = {}  # {label: list[float]}     — courbes epsilon


# ==================================================================
# Run 1 — Brute Force
# ==================================================================

def run_brute_force():
    print("\n" + "="*55)
    print("  RUN 1 — Brute Force (baseline naïf)")
    print("="*55)
    env   = TaxiWrapper(reward_mode="default")
    agent = BruteForceAgent(env, seed=0)
    tracker = agent.test(n_episodes=N_TEST, log_interval=50, seed=TEST_SEED)
    test_trackers["BruteForce"] = tracker
    env.close()


# ==================================================================
# Run 2 — Q-Learning non-optimisé
# ==================================================================

def run_ql_default():
    print("\n" + "="*55)
    print("  RUN 2 — Q-Learning (paramètres par défaut)")
    print("="*55)
    env   = TaxiWrapper(reward_mode="default")
    agent = QLearningAgent(
        env,
        alpha=0.1, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
        seed=0,
    )
    print(agent)

    train = agent.train(n_episodes=1000, log_interval=200)
    test  = agent.test(n_episodes=N_TEST, log_interval=50, seed=TEST_SEED)
    agent.save(os.path.join(MODELS_DIR, "ql_default.pkl"))

    train_trackers["Q-Learning (default)"]    = train
    test_trackers["Q-Learning (default)"]     = test
    epsilon_histories["Q-Learning (default)"] = agent.epsilon_history
    env.close()


# ==================================================================
# Run 3 — Q-Learning optimisé
# ==================================================================

def run_ql_tuned():
    print("\n" + "="*55)
    print("  RUN 3 — Q-Learning (paramètres optimisés)")
    print("="*55)
    env   = TaxiWrapper(reward_mode="default")
    agent = QLearningAgent(
        env,
        alpha=0.8, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01,
        seed=0,
    )
    print(agent)

    train = agent.train(n_episodes=10_000, log_interval=1000)
    test  = agent.test(n_episodes=N_TEST, log_interval=50, seed=TEST_SEED)
    agent.save(os.path.join(MODELS_DIR, "ql_tuned.pkl"))

    train_trackers["Q-Learning (tuned)"]    = train
    test_trackers["Q-Learning (tuned)"]     = test
    epsilon_histories["Q-Learning (tuned)"] = agent.epsilon_history
    env.close()


# ==================================================================
# Run 4 — DQN non-optimisé
# ==================================================================

def run_dqn_default():
    print("\n" + "="*55)
    print("  RUN 4 — DQN (paramètres par défaut)")
    print("="*55)
    env   = TaxiWrapper(reward_mode="default")
    agent = DQNAgent(
        env,
        lr=0.001, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
        batch_size=64, buffer_capacity=10_000, target_update_freq=100,
        seed=0,
    )
    print(agent)

    train = agent.train(n_episodes=3000, log_interval=300)
    test  = agent.test(n_episodes=N_TEST, log_interval=50, seed=TEST_SEED)
    agent.save(os.path.join(MODELS_DIR, "dqn_default.pt"))

    train_trackers["DQN (default)"]    = train
    test_trackers["DQN (default)"]     = test
    epsilon_histories["DQN (default)"] = agent.epsilon_history
    env.close()


# ==================================================================
# Run 5 — DQN optimisé
# ==================================================================

def run_dqn_tuned():
    print("\n" + "="*55)
    print("  RUN 5 — DQN (paramètres optimisés)")
    print("="*55)
    env   = TaxiWrapper(reward_mode="default")
    agent = DQNAgent(
        env,
        lr=0.0005, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.01,
        batch_size=64, buffer_capacity=20_000, target_update_freq=200,
        seed=0,
    )
    print(agent)

    train = agent.train(n_episodes=10_000, log_interval=1000)
    test  = agent.test(n_episodes=N_TEST, log_interval=50, seed=TEST_SEED)
    agent.save(os.path.join(MODELS_DIR, "dqn_tuned.pt"))

    train_trackers["DQN (tuned)"]    = train
    test_trackers["DQN (tuned)"]     = test
    epsilon_histories["DQN (tuned)"] = agent.epsilon_history
    env.close()


# ==================================================================
# Run 6 — Reward Shaping comparison (sur Q-Learning tuned)
# ==================================================================

def run_reward_shaping():
    print("\n" + "="*55)
    print("  RUN 6 — Comparaison Reward Shaping")
    print("="*55)

    shaping_trackers = {}

    for mode in ["default", "distance"]:
        print(f"\n  → reward_mode='{mode}'")
        env   = TaxiWrapper(reward_mode=mode)
        agent = QLearningAgent(
            env,
            alpha=0.8, gamma=0.99,
            epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01,
            seed=0,
        )
        train = agent.train(n_episodes=5000, log_interval=1000)
        shaping_trackers[f"Q-Learning ({mode})"] = train
        env.close()

    plot_reward_shaping_comparison(shaping_trackers)


# ==================================================================
# Run 7 — Heatmap hyperparamètres (alpha × gamma sur Q-Learning)
# ==================================================================

def run_hyperparam_search():
    print("\n" + "="*55)
    print("  RUN 7 — Recherche hyperparamètres (alpha × gamma)")
    print("="*55)

    alphas = [0.1, 0.3, 0.5, 0.8]
    gammas = [0.90, 0.95, 0.99]
    heatmap_results = {}

    for alpha in alphas:
        for gamma in gammas:
            print(f"  α={alpha}  γ={gamma} ...", end="", flush=True)
            env   = TaxiWrapper(reward_mode="default")
            agent = QLearningAgent(
                env, alpha=alpha, gamma=gamma,
                epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, seed=0,
            )
            agent.train(n_episodes=5000, log_interval=999_999)
            test = agent.test(n_episodes=50, log_interval=999_999, seed=TEST_SEED)
            s = test.summary()
            heatmap_results[(alpha, gamma)] = s
            print(f"  mean_steps={s['mean_steps']:.1f}")
            env.close()

    plot_hyperparam_heatmap(heatmap_results, param_x="alpha", param_y="gamma", metric="mean_steps")
    plot_hyperparam_heatmap(heatmap_results, param_x="alpha", param_y="gamma", metric="success_rate")


# ==================================================================
# Graphiques comparatifs finaux
# ==================================================================

def generate_final_plots():
    print("\n" + "="*55)
    print("  GRAPHIQUES COMPARATIFS FINAUX")
    print("="*55)

    # Courbes d'apprentissage (training) — tous sauf brute force
    if train_trackers:
        plot_learning_curves(train_trackers)
        plot_steps_evolution(train_trackers)

    # Décroissance epsilon
    if epsilon_histories:
        plot_epsilon_decay(epsilon_histories)

    # Distribution des steps en test — tous les algorithmes
    plot_steps_distribution(test_trackers)

    # Tableau récapitulatif
    summaries = {label: t.summary() for label, t in test_trackers.items()}
    plot_summary_table(summaries)


# ==================================================================
# Point d'entrée
# ==================================================================

def main():
    print("\n" + "="*55)
    print("  TAXI DRIVER — BENCHMARK COMPLET")
    print(f"  Test : {N_TEST} épisodes | seed={TEST_SEED}")
    print("="*55)

    run_brute_force()
    run_ql_default()
    run_ql_tuned()
    run_dqn_default()
    run_dqn_tuned()
    run_reward_shaping()
    run_hyperparam_search()
    generate_final_plots()

    print("\n" + "="*55)
    print("  Benchmark terminé. Graphiques dans results/plots/")
    print("="*55)


if __name__ == "__main__":
    main()

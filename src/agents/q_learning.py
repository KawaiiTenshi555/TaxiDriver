# q_learning.py — Agent Q-Learning tabulaire (baseline RL)

import os
import pickle

import numpy as np
from tqdm import tqdm

from environment.taxi_wrapper import TaxiWrapper
from utils.metrics import MetricsTracker
from utils.replay import run_console_replay


class QLearningAgent:
    """
    Agent Q-Learning tabulaire, off-policy, model-free.

    Maintient une Q-table de taille (n_states × n_actions) mise à jour
    à chaque step via la règle de Bellman :

        Q(s, a) ← Q(s, a) + α × [r + γ × max_a' Q(s', a') − Q(s, a)]

    Exploration : politique epsilon-greedy avec décroissance exponentielle de ε.
    """

    def __init__(
        self,
        env: TaxiWrapper,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed: int = None,
    ):
        """
        Args:
            env           (TaxiWrapper): environnement Taxi-v3 wrappé
            alpha         (float)      : taux d'apprentissage (0 < α ≤ 1)
            gamma         (float)      : facteur de discount (0 < γ ≤ 1)
            epsilon       (float)      : taux d'exploration initial (0 ≤ ε ≤ 1)
            epsilon_decay (float)      : multiplicateur de décroissance de ε par épisode
            epsilon_min   (float)      : valeur plancher de ε
            seed          (int | None) : graine aléatoire
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.default_rng(seed)

        # Q-table initialisée à zéro : shape (500, 6)
        self.q_table = np.zeros((env.n_states, env.n_actions), dtype=np.float64)

        # Historique de ε pour la visualisation
        self.epsilon_history: list[float] = []

    # ------------------------------------------------------------------
    # Sélection d'action
    # ------------------------------------------------------------------

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Politique epsilon-greedy pendant l'entraînement, greedy pendant le test.

        Args:
            state    (int) : état courant (0-499)
            training (bool): True = epsilon-greedy, False = greedy pur

        Returns:
            int: action choisie (0-5)
        """
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.env.n_actions))
        return int(np.argmax(self.q_table[state]))

    # ------------------------------------------------------------------
    # Mise à jour de la Q-table (règle de Bellman)
    # ------------------------------------------------------------------

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """
        Applique la mise à jour Q-Learning sur une transition (s, a, r, s').

        La cible est bootstrappée sur la valeur maximale de l'état suivant,
        sauf si l'épisode est terminé (done=True), auquel cas la valeur
        future est nulle.

        Args:
            state      (int)  : état courant
            action     (int)  : action exécutée
            reward     (float): récompense reçue
            next_state (int)  : état suivant
            done       (bool) : True si l'épisode est terminé (terminated ou truncated)
        """
        current_q = self.q_table[state, action]
        target = reward + (0.0 if done else self.gamma * np.max(self.q_table[next_state]))
        self.q_table[state, action] += self.alpha * (target - current_q)

    # ------------------------------------------------------------------
    # Décroissance d'epsilon
    # ------------------------------------------------------------------

    def _decay_epsilon(self) -> None:
        """Applique la décroissance exponentielle de ε après chaque épisode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)

    # ------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------

    def train(self, n_episodes: int, log_interval: int = 100, seed: int = None) -> MetricsTracker:
        """
        Boucle d'entraînement complète.

        À chaque épisode :
          1. Reset de l'environnement
          2. Boucle step : select_action → env.step → update Q-table
          3. Décroissance de ε

        Args:
            n_episodes   (int): nombre d'épisodes d'entraînement
            log_interval (int): fréquence d'affichage console
            seed         (int): graine pour env.reset (reproductibilité)

        Returns:
            MetricsTracker: données de tous les épisodes d'entraînement
        """
        tracker = MetricsTracker(n_episodes=n_episodes, log_interval=log_interval, phase="Training")

        for ep in tqdm(range(n_episodes), desc="Q-Learning Training", unit="ep", leave=False):
            state, _ = self.env.reset(seed=seed)
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            tracker.begin_episode(ep)

            while not (terminated or truncated):
                action = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                done = terminated or truncated
                self.update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps += 1

            self._decay_epsilon()
            tracker.end_episode(
                episode=ep,
                steps=steps,
                total_reward=total_reward,
                success=terminated,
            )

        return tracker

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test(self, n_episodes: int, log_interval: int = 50, seed: int = None) -> MetricsTracker:
        """
        Évalue la politique apprise (greedy pure, ε=0) sur n_episodes.

        Args:
            n_episodes   (int): nombre d'épisodes de test
            log_interval (int): fréquence d'affichage console
            seed         (int): graine pour env.reset

        Returns:
            MetricsTracker: données de tous les épisodes de test
        """
        tracker = MetricsTracker(n_episodes=n_episodes, log_interval=log_interval, phase="Test")

        for ep in tqdm(range(n_episodes), desc="Q-Learning Test", unit="ep", leave=False):
            state, _ = self.env.reset(seed=seed)
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            tracker.begin_episode(ep)

            while not (terminated or truncated):
                action = self.select_action(state, training=False)
                state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                steps += 1

            tracker.end_episode(
                episode=ep,
                steps=steps,
                total_reward=total_reward,
                success=terminated,
            )

        tracker.print_summary(label="=== Q-Learning — Résultats Test ===")
        return tracker

    # ------------------------------------------------------------------
    # Sauvegarde / chargement
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Sauvegarde la Q-table et les hyperparamètres sur disque.

        Args:
            path (str): chemin du fichier de sauvegarde (.pkl)
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "q_table": self.q_table,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "epsilon_history": self.epsilon_history,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[QLearning] Modèle sauvegardé → {path}")

    def load(self, path: str) -> None:
        """
        Charge une Q-table et les hyperparamètres depuis un fichier .pkl.

        Args:
            path (str): chemin du fichier de sauvegarde (.pkl)
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.q_table = payload["q_table"]
        self.alpha = payload["alpha"]
        self.gamma = payload["gamma"]
        self.epsilon = payload["epsilon"]
        self.epsilon_decay = payload["epsilon_decay"]
        self.epsilon_min = payload["epsilon_min"]
        self.epsilon_history = payload.get("epsilon_history", [])
        print(f"[QLearning] Modèle chargé ← {path}")

    # ------------------------------------------------------------------
    # Replay d'épisodes (affichage console)
    # ------------------------------------------------------------------

    def play_episode(self, seed: int = None) -> None:
        """Replay one full episode with learned greedy policy."""
        run_console_replay(
            env=self.env,
            policy=lambda state: self.select_action(state, training=False),
            title="Q-Learning (greedy)",
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Représentation
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"QLearningAgent(α={self.alpha}, γ={self.gamma}, "
            f"ε={self.epsilon:.4f}, decay={self.epsilon_decay})"
        )

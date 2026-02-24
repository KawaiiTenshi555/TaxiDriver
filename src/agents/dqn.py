# dqn.py — Agent Deep Q-Network (algorithme principal optimisé)

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from environment.taxi_wrapper import TaxiWrapper
from utils.metrics import MetricsTracker
from utils.replay_buffer import ReplayBuffer
from utils.replay import run_console_replay


# ==================================================================
# Réseau de neurones — approximateur de Q-valeurs
# ==================================================================

class QNetwork(nn.Module):
    """
    Réseau fully-connected qui approche Q(s, a) pour toutes les actions.

    Architecture :
        Input  (500,)  → Linear → ReLU → Linear → ReLU → Linear → Output (6,)

    Input  : vecteur one-hot de l'état (taille n_states = 500)
    Output : Q-valeurs pour chacune des 6 actions
    """

    def __init__(self, n_states: int, n_actions: int, hidden_1: int = 128, hidden_2: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states,  hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1,  hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2,  n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==================================================================
# Agent DQN
# ==================================================================

class DQNAgent:
    """
    Agent Deep Q-Network (DQN) avec Experience Replay et Target Network.

    Composants clés :
    - QNetwork         : réseau principal mis à jour à chaque step
    - Target Network   : copie du réseau principal, mis à jour toutes les
                         target_update_freq étapes pour stabiliser les cibles
    - ReplayBuffer     : stocke les transitions et fournit des mini-batches
                         aléatoires pour briser la corrélation temporelle
    - Epsilon-greedy   : exploration décroissante

    Règle de mise à jour :
        L = MSE( Q(s,a) ,  r + γ · max_a' Q_target(s', a') · (1 - done) )
    """

    def __init__(
        self,
        env: TaxiWrapper,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 64,
        buffer_capacity: int = 10_000,
        target_update_freq: int = 100,
        hidden_1: int = 128,
        hidden_2: int = 64,
        seed: int = None,
    ):
        """
        Args:
            env                (TaxiWrapper): environnement wrappé
            lr                 (float)      : learning rate Adam
            gamma              (float)      : facteur de discount
            epsilon            (float)      : taux d'exploration initial
            epsilon_decay      (float)      : décroissance de ε par épisode
            epsilon_min        (float)      : valeur plancher de ε
            batch_size         (int)        : taille du mini-batch
            buffer_capacity    (int)        : capacité du replay buffer
            target_update_freq (int)        : fréquence de sync target network (en steps)
            hidden_1           (int)        : neurones couche cachée 1
            hidden_2           (int)        : neurones couche cachée 2
            seed               (int | None) : graine aléatoire
        """
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.rng = np.random.default_rng(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Réseaux
        self.q_net = QNetwork(env.n_states, env.n_actions, hidden_1, hidden_2).to(self.device)
        self.target_net = QNetwork(env.n_states, env.n_actions, hidden_1, hidden_2).to(self.device)
        self._sync_target()

        # Optimiseur et loss
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

        # Compteurs et historiques
        self._step_count = 0
        self.epsilon_history: list[float] = []
        self.loss_history: list[float] = []

    # ------------------------------------------------------------------
    # Synchronisation target network
    # ------------------------------------------------------------------

    def _sync_target(self) -> None:
        """Copie les poids du réseau principal vers le target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ------------------------------------------------------------------
    # Encodage de l'état
    # ------------------------------------------------------------------

    def _encode(self, state: int) -> torch.Tensor:
        """
        Convertit un état entier en tenseur one-hot pour le réseau.

        Returns:
            torch.Tensor float32 de forme (1, n_states)
        """
        one_hot = self.env.get_state_features(state)
        return torch.tensor(one_hot, dtype=torch.float32, device=self.device).unsqueeze(0)

    # ------------------------------------------------------------------
    # Sélection d'action
    # ------------------------------------------------------------------

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Politique epsilon-greedy en training, greedy en test.

        Args:
            state    (int) : état courant
            training (bool): True = epsilon-greedy, False = greedy pur

        Returns:
            int: action choisie (0-5)
        """
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.env.n_actions))

        with torch.no_grad():
            q_values = self.q_net(self._encode(state))
        return int(q_values.argmax().item())

    # ------------------------------------------------------------------
    # Apprentissage (mise à jour du réseau)
    # ------------------------------------------------------------------

    def learn(self) -> float | None:
        """
        Tire un mini-batch depuis le replay buffer et effectue
        une étape de descente de gradient sur le réseau principal.

        Returns:
            float | None : valeur de la loss, ou None si le buffer
                           n'est pas encore assez rempli
        """
        if not self.buffer.is_ready(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size, self.rng
        )

        # Conversion en tenseurs one-hot
        states_t      = torch.zeros((self.batch_size, self.env.n_states),
                                    dtype=torch.float32, device=self.device)
        next_states_t = torch.zeros_like(states_t)
        for i, (s, ns) in enumerate(zip(states, next_states)):
            states_t[i]      = torch.tensor(self.env.get_state_features(s))
            next_states_t[i] = torch.tensor(self.env.get_state_features(ns))

        actions_t = torch.tensor(actions, dtype=torch.int64,   device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t   = torch.tensor(dones,   dtype=torch.float32, device=self.device)

        # Q-valeurs courantes pour l'action réellement prise
        current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Cibles via target network
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1).values
            targets = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Décroissance d'epsilon
    # ------------------------------------------------------------------

    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)

    # ------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------

    def train(self, n_episodes: int, log_interval: int = 100, seed: int = None) -> MetricsTracker:
        """
        Boucle d'entraînement DQN complète.

        À chaque step :
          1. select_action (epsilon-greedy)
          2. env.step
          3. buffer.push
          4. learn() si buffer suffisamment rempli
          5. sync target_net tous les target_update_freq steps

        Args:
            n_episodes   (int): nombre d'épisodes d'entraînement
            log_interval (int): fréquence d'affichage console
            seed         (int): graine pour env.reset

        Returns:
            MetricsTracker
        """
        tracker = MetricsTracker(n_episodes=n_episodes, log_interval=log_interval, phase="Training")

        for ep in tqdm(range(n_episodes), desc="DQN Training", unit="ep", leave=False):
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
                self.buffer.push(state, action, reward, next_state, done)

                loss = self.learn()
                if loss is not None:
                    self.loss_history.append(loss)

                self._step_count += 1
                if self._step_count % self.target_update_freq == 0:
                    self._sync_target()

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
        Évalue la politique apprise (greedy pure) sur n_episodes.

        Args:
            n_episodes   (int): nombre d'épisodes de test
            log_interval (int): fréquence d'affichage console
            seed         (int): graine pour env.reset

        Returns:
            MetricsTracker
        """
        tracker = MetricsTracker(n_episodes=n_episodes, log_interval=log_interval, phase="Test")

        for ep in tqdm(range(n_episodes), desc="DQN Test", unit="ep", leave=False):
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

        tracker.print_summary(label="=== DQN — Résultats Test ===")
        return tracker

    # ------------------------------------------------------------------
    # Sauvegarde / chargement
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Sauvegarde les poids du réseau et les hyperparamètres.

        Args:
            path (str): chemin du fichier (.pt)
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "q_net_state":      self.q_net.state_dict(),
            "target_net_state": self.target_net.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "hyperparams": {
                "lr": self.lr,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_min": self.epsilon_min,
                "batch_size": self.batch_size,
                "target_update_freq": self.target_update_freq,
            },
            "epsilon_history": self.epsilon_history,
        }, path)
        print(f"[DQN] Modèle sauvegardé → {path}")

    def load(self, path: str) -> None:
        """
        Charge les poids et hyperparamètres depuis un fichier .pt.

        Args:
            path (str): chemin du fichier
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net_state"])
        self.target_net.load_state_dict(checkpoint["target_net_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        hp = checkpoint["hyperparams"]
        self.lr             = hp["lr"]
        self.gamma          = hp["gamma"]
        self.epsilon        = hp["epsilon"]
        self.epsilon_decay  = hp["epsilon_decay"]
        self.epsilon_min    = hp["epsilon_min"]
        self.batch_size     = hp["batch_size"]
        self.target_update_freq = hp["target_update_freq"]
        self.epsilon_history = checkpoint.get("epsilon_history", [])
        print(f"[DQN] Modèle chargé ← {path}")

    # ------------------------------------------------------------------
    # Replay d'épisodes (affichage console)
    # ------------------------------------------------------------------

    def play_episode(self, seed: int = None) -> None:
        """Replay one full episode with learned greedy policy."""
        run_console_replay(
            env=self.env,
            policy=lambda state: self.select_action(state, training=False),
            title="DQN (greedy)",
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Représentation
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"DQNAgent(lr={self.lr}, γ={self.gamma}, "
            f"ε={self.epsilon:.4f}, batch={self.batch_size}, "
            f"target_freq={self.target_update_freq}, device={self.device})"
        )

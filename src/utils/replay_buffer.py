# replay_buffer.py — Experience Replay Buffer pour DQN

from collections import deque

import numpy as np


class ReplayBuffer:
    """
    Buffer circulaire qui stocke les transitions (s, a, r, s', done)
    et permet de tirer des mini-batches aléatoires pour l'entraînement DQN.

    Rôle clé : briser la corrélation temporelle entre les transitions
    consécutives, ce qui stabilise la convergence du réseau de neurones.
    """

    def __init__(self, capacity: int = 10_000):
        """
        Args:
            capacity (int): nombre maximal de transitions stockées.
                            Les plus anciennes sont écrasées automatiquement.
        """
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)

    # ------------------------------------------------------------------
    # Stockage
    # ------------------------------------------------------------------

    def push(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """
        Ajoute une transition dans le buffer.

        Args:
            state      (int)  : état courant
            action     (int)  : action exécutée
            reward     (float): récompense reçue
            next_state (int)  : état suivant
            done       (bool) : True si l'épisode est terminé
        """
        self._buffer.append((state, action, reward, next_state, done))

    # ------------------------------------------------------------------
    # Échantillonnage
    # ------------------------------------------------------------------

    def sample(self, batch_size: int, rng: np.random.Generator):
        """
        Tire un mini-batch aléatoire sans remise depuis le buffer.

        Args:
            batch_size (int)                : taille du batch
            rng        (np.random.Generator): générateur aléatoire

        Returns:
            Tuple de 5 np.ndarray :
                states      (int32)   shape (batch_size,)
                actions     (int64)   shape (batch_size,)
                rewards     (float32) shape (batch_size,)
                next_states (int32)   shape (batch_size,)
                dones       (float32) shape (batch_size,)  0.0 ou 1.0
        """
        indices = rng.choice(len(self._buffer), size=batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states,      dtype=np.int32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.int32),
            np.array(dones,       dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Retourne True si le buffer contient assez de transitions pour un batch."""
        return len(self._buffer) >= batch_size

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self._buffer)}/{self.capacity})"

# taxi_wrapper.py — Wrapper Gymnasium pour Taxi-v3 + reward shaping

import gymnasium as gym
import numpy as np

# Positions des 4 emplacements désignés sur la grille 5x5
# Index : 0=R (Red), 1=G (Green), 2=Y (Yellow), 3=B (Blue)
LOCATIONS = {
    0: (0, 0),  # R
    1: (0, 4),  # G
    2: (4, 0),  # Y
    3: (4, 3),  # B
}

REWARD_MODES = ["default", "distance", "custom"]


class TaxiWrapper:
    """
    Wrapper autour de l'environnement Gymnasium Taxi-v3.

    Centralise toutes les interactions avec l'environnement et permet
    d'injecter différentes stratégies de reward shaping sans toucher
    au code des agents.

    Espace des états  : 500 états discrets (25 positions taxi × 5 positions
                        passager × 4 destinations)
    Espace des actions: 6 actions discrètes
                        0=Sud, 1=Nord, 2=Est, 3=Ouest, 4=Pickup, 5=Dropoff
    Récompenses défaut: -1 par step, +20 dropoff réussi, -10 action illégale
    """

    def __init__(self, reward_mode="default", custom_rewards=None, render_mode=None):
        """
        Args:
            reward_mode (str)    : "default" | "distance" | "custom"
            custom_rewards (dict): paramètres de récompenses si reward_mode != "default"
                Clés reconnues :
                    "step"                 (float) pénalité par étape
                    "dropoff_success"      (float) bonus dépôt réussi
                    "illegal_action"       (float) pénalité action illégale
                    "distance_bonus_scale" (float) facteur du bonus de distance
            render_mode (str)    : None | "ansi" | "human"
        """
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode doit être parmi {REWARD_MODES}")

        self.reward_mode = reward_mode
        self.render_mode = render_mode

        self._env = gym.make("Taxi-v3", render_mode=render_mode)

        # Dimensions de l'espace d'états et d'actions
        self.n_states = self._env.observation_space.n   # 500
        self.n_actions = self._env.action_space.n       # 6

        # Configuration des récompenses
        self._rewards = self._build_reward_config(custom_rewards)

        # État interne pour le reward shaping basé sur la distance
        self._prev_distance = None
        self._passenger_in_taxi = False

    # ------------------------------------------------------------------
    # Configuration des récompenses
    # ------------------------------------------------------------------

    def _build_reward_config(self, custom_rewards):
        """
        Construit la configuration des récompenses.
        Les valeurs custom écrasent les valeurs par défaut.
        """
        config = {
            "step": -1,
            "dropoff_success": +20,
            "illegal_action": -10,
            "distance_bonus_scale": 1.0,
        }
        if custom_rewards:
            config.update(custom_rewards)
        return config

    # ------------------------------------------------------------------
    # API principale (reset / step / render / close)
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        """
        Réinitialise l'environnement pour un nouvel épisode.

        Returns:
            state (int)  : état initial (entier 0-499)
            info  (dict) : informations supplémentaires de Gymnasium
        """
        state, info = self._env.reset(seed=seed)
        self._passenger_in_taxi = False
        self._prev_distance = self._compute_relevant_distance(state)
        return state, info

    def step(self, action):
        """
        Effectue une action dans l'environnement.

        Args:
            action (int): action à exécuter (0-5)

        Returns:
            next_state  (int)  : nouvel état
            reward      (float): récompense (potentiellement shapée)
            terminated  (bool) : True si l'épisode est terminé avec succès
            truncated   (bool) : True si le nombre max de steps est atteint
            info        (dict) : informations supplémentaires
        """
        next_state, reward, terminated, truncated, info = self._env.step(action)

        if self.reward_mode == "distance":
            reward = self._shaped_reward_distance(next_state, reward, terminated)
        elif self.reward_mode == "custom":
            reward = self._shaped_reward_custom(next_state, reward, terminated)

        self._prev_distance = self._compute_relevant_distance(next_state)
        return next_state, reward, terminated, truncated, info

    def render(self):
        """Affiche l'état courant de l'environnement en console (ASCII)."""
        return self._env.render()

    def close(self):
        """Libère les ressources de l'environnement."""
        self._env.close()

    # ------------------------------------------------------------------
    # Décodage et encodage de l'état
    # ------------------------------------------------------------------

    def decode_state(self, state):
        """
        Décompose un entier d'état en composantes lisibles.

        Args:
            state (int): entier d'état (0-499)

        Returns:
            dict avec les clés :
                "taxi_row"      (int): ligne du taxi (0-4)
                "taxi_col"      (int): colonne du taxi (0-4)
                "passenger_loc" (int): position du passager
                                       0=R, 1=G, 2=Y, 3=B, 4=dans le taxi
                "destination"   (int): destination cible (0=R, 1=G, 2=Y, 3=B)
        """
        taxi_row, taxi_col, passenger_loc, destination = self._env.unwrapped.decode(state)
        return {
            "taxi_row": int(taxi_row),
            "taxi_col": int(taxi_col),
            "passenger_loc": int(passenger_loc),
            "destination": int(destination),
        }

    def get_state_features(self, state):
        """
        Encode l'état en vecteur one-hot pour le DQN.

        Args:
            state (int): entier d'état (0-499)

        Returns:
            np.ndarray float32 de forme (500,)
        """
        one_hot = np.zeros(self.n_states, dtype=np.float32)
        one_hot[state] = 1.0
        return one_hot

    # ------------------------------------------------------------------
    # Reward shaping — distance
    # ------------------------------------------------------------------

    def _compute_relevant_distance(self, state):
        """
        Calcule la distance Manhattan pertinente selon la phase de l'épisode :
            - Phase pickup  : distance taxi → passager
            - Phase dropoff : distance taxi → destination
        """
        decoded = self.decode_state(state)
        taxi_pos = (decoded["taxi_row"], decoded["taxi_col"])

        if decoded["passenger_loc"] == 4:
            # Passager dans le taxi → on mesure la distance vers la destination
            self._passenger_in_taxi = True
            dest_pos = LOCATIONS[decoded["destination"]]
            return self._manhattan(taxi_pos, dest_pos)
        else:
            # Passager pas encore récupéré → distance vers le passager
            self._passenger_in_taxi = False
            pass_pos = LOCATIONS[decoded["passenger_loc"]]
            return self._manhattan(taxi_pos, pass_pos)

    def _shaped_reward_distance(self, next_state, original_reward, terminated):
        """
        Reward shaping basé sur la réduction de distance.

        Les récompenses clés (dropoff réussi, action illégale) sont conservées
        telles quelles. Le bonus de distance s'applique uniquement aux steps
        neutres pour guider l'exploration.
        """
        if terminated and original_reward == 20:
            return self._rewards["dropoff_success"]
        if original_reward == -10:
            return self._rewards["illegal_action"]

        new_distance = self._compute_relevant_distance(next_state)
        distance_improvement = self._prev_distance - new_distance
        bonus = distance_improvement * self._rewards["distance_bonus_scale"]

        return self._rewards["step"] + bonus

    def _shaped_reward_custom(self, next_state, original_reward, terminated):
        """Mapping entièrement custom des récompenses (hors distance)."""
        if terminated and original_reward == 20:
            return self._rewards["dropoff_success"]
        if original_reward == -10:
            return self._rewards["illegal_action"]
        return self._rewards["step"]

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    @staticmethod
    def _manhattan(pos_a, pos_b):
        """Distance Manhattan entre deux positions (row, col)."""
        return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def __repr__(self):
        return (
            f"TaxiWrapper(reward_mode='{self.reward_mode}', "
            f"n_states={self.n_states}, n_actions={self.n_actions})"
        )

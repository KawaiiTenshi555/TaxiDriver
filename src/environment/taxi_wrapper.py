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
    Wrapper autour de l'environnement Gymnasium Taxi-v3 (défaut) ou d'un
    environnement custom (grid_size != 5 ou n_passengers != 1).

    Centralise toutes les interactions avec l'environnement et permet
    d'injecter différentes stratégies de reward shaping sans toucher
    au code des agents.

    Espace des états  : 500 états discrets pour l'env original (5×5, 1 passager)
                        ou calculé automatiquement pour les envs custom.
    Espace des actions: 6 actions discrètes
                        0=Sud, 1=Nord, 2=Est, 3=Ouest, 4=Pickup, 5=Dropoff
    Récompenses défaut: -1 par step, +20 dropoff réussi, -10 action illégale
    """

    def __init__(
        self,
        reward_mode="default",
        custom_rewards=None,
        render_mode=None,
        grid_size: int = 5,
        n_passengers: int = 1,
    ):
        """
        Args:
            reward_mode   (str)    : "default" | "distance" | "custom"
            custom_rewards (dict)  : paramètres de récompenses si reward_mode != "default"
            render_mode   (str)    : None | "ansi" | "human" (ignoré pour env custom)
            grid_size     (int)    : taille de la grille (3-12). Défaut 5.
            n_passengers  (int)    : nombre de passagers (1-7). Défaut 1.
        """
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode doit être parmi {REWARD_MODES}")

        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self.grid_size = grid_size
        self.n_passengers = n_passengers

        # Détermine si on utilise l'env Gymnasium original ou l'env custom
        self._custom_mode = not (grid_size == 5 and n_passengers == 1)

        if self._custom_mode:
            from environment.custom_taxi import CustomTaxiEnv
            _cr = None
            if reward_mode == "custom":
                _cr = custom_rewards
            self._custom_env = CustomTaxiEnv(
                grid_size=grid_size,
                n_passengers=n_passengers,
                reward_mode=reward_mode,
                custom_rewards=_cr,
            )
            self.n_states   = self._custom_env.n_states
            self.n_actions  = self._custom_env.n_actions
            self._env       = None
            self._rewards   = self._custom_env._rewards
        else:
            self._custom_env = None
            self._env = gym.make("Taxi-v3", render_mode=render_mode)
            self.n_states  = self._env.observation_space.n   # 500
            self.n_actions = self._env.action_space.n        # 6
            self._rewards  = self._build_reward_config(custom_rewards)

        # État interne pour le reward shaping basé sur la distance (env original seulement)
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
            info  (dict) : informations supplémentaires
        """
        if self._custom_mode:
            return self._custom_env.reset(seed=seed)
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
        if self._custom_mode:
            return self._custom_env.step(action)
        next_state, reward, terminated, truncated, info = self._env.step(action)

        if self.reward_mode == "distance":
            reward = self._shaped_reward_distance(next_state, reward, terminated)
        elif self.reward_mode == "custom":
            reward = self._shaped_reward_custom(next_state, reward, terminated)

        self._prev_distance = self._compute_relevant_distance(next_state)
        return next_state, reward, terminated, truncated, info

    def render(self):
        """Affiche l'état courant de l'environnement en console (ASCII)."""
        if self._custom_mode:
            return self._custom_env.render()
        return self._env.render()

    def close(self):
        """Libère les ressources de l'environnement."""
        if self._custom_mode:
            self._custom_env.close()
        elif self._env is not None:
            self._env.close()

    # ------------------------------------------------------------------
    # Décodage et encodage de l'état
    # ------------------------------------------------------------------

    def decode_state(self, state):
        """
        Décompose un entier d'état en composantes lisibles.

        Pour l'env original (Taxi-v3), retourne un dict avec les clés :
            "taxi_row", "taxi_col", "passenger_loc", "destination"
            ainsi que "passengers" (format unifié).
        Pour l'env custom, retourne le format unifié avec "passengers".
        """
        if self._custom_mode:
            return self._custom_env.decode_state(state)
        taxi_row, taxi_col, passenger_loc, destination = self._env.unwrapped.decode(state)
        # Format unifié pour la GUI
        return {
            "taxi_row":      int(taxi_row),
            "taxi_col":      int(taxi_col),
            "passenger_loc": int(passenger_loc),   # 0-3 = lieu, 4 = dans le taxi
            "destination":   int(destination),
            "grid_size":     5,
            "locations":     list(LOCATIONS.values()),
            "passengers": [
                {
                    "status":     1 if int(passenger_loc) == 4 else 0,
                    "pickup_loc": int(passenger_loc) if int(passenger_loc) < 4 else -1,
                    "dest":       int(destination),
                }
            ],
        }

    def get_state_features(self, state):
        """
        Encode l'état en vecteur de features pour le DQN.

        Pour l'env original : vecteur one-hot de taille n_states (500).
        Pour l'env custom   : vecteur compact de taille n_state_features.
        """
        if self._custom_mode:
            return self._custom_env.get_state_features(state)
        one_hot = np.zeros(self.n_states, dtype=np.float32)
        one_hot[state] = 1.0
        return one_hot

    @property
    def n_state_features(self) -> int:
        """Taille du vecteur de features pour le DQN."""
        if self._custom_mode:
            return self._custom_env.n_state_features
        return self.n_states  # 500 pour l'env original

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
        if self._custom_mode:
            return repr(self._custom_env)
        return (
            f"TaxiWrapper(reward_mode='{self.reward_mode}', "
            f"n_states={self.n_states}, n_actions={self.n_actions})"
        )

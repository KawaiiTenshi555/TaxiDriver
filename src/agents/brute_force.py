# brute_force.py — Agent aléatoire (baseline naïf)

import numpy as np
from tqdm import tqdm

from environment.taxi_wrapper import TaxiWrapper
from utils.metrics import MetricsTracker


class BruteForceAgent:
    """
    Agent aléatoire pur : choisit une action uniformément au hasard
    à chaque step, sans aucun apprentissage ni mémoire entre les épisodes.

    Sert de borne inférieure de performance (~350 steps en moyenne).
    Référence attendue par le sujet pour valider les gains des agents RL.
    """

    def __init__(self, env: TaxiWrapper, seed=None):
        """
        Args:
            env  (TaxiWrapper): environnement Taxi-v3 wrappé
            seed (int | None) : graine aléatoire pour la reproductibilité
        """
        self.env = env
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Sélection d'action
    # ------------------------------------------------------------------

    def select_action(self, state=None):
        """
        Retourne une action aléatoire uniforme parmi les 6 actions.
        Le paramètre state est accepté mais ignoré (pas d'apprentissage).
        """
        return int(self.rng.integers(0, self.env.n_actions))

    # ------------------------------------------------------------------
    # Boucle de run (training = test ici, aucun apprentissage)
    # ------------------------------------------------------------------

    def _run_episodes(self, n_episodes, phase, log_interval, seed):
        """
        Boucle générique d'exécution des épisodes.

        Args:
            n_episodes   (int): nombre d'épisodes à exécuter
            phase        (str): libellé pour les logs ("Test")
            log_interval (int): fréquence d'affichage console
            seed         (int): graine pour reset de l'environnement

        Returns:
            MetricsTracker: tracker rempli avec les données de tous les épisodes
        """
        tracker = MetricsTracker(n_episodes=n_episodes, log_interval=log_interval, phase=phase)

        for ep in tqdm(range(n_episodes), desc=phase, unit="ep", leave=False):
            state, _ = self.env.reset(seed=seed)
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            tracker.begin_episode(ep)

            while not (terminated or truncated):
                action = self.select_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                steps += 1

            tracker.end_episode(
                episode=ep,
                steps=steps,
                total_reward=total_reward,
                success=terminated,   # terminated=True uniquement sur dropoff réussi
            )

        return tracker

    def test(self, n_episodes=100, log_interval=50, seed=None):
        """
        Exécute n_episodes avec une politique aléatoire et retourne les métriques.

        Args:
            n_episodes   (int): nombre d'épisodes de test
            log_interval (int): fréquence d'affichage console
            seed         (int): graine pour la reproductibilité

        Returns:
            MetricsTracker: tracker avec toutes les données de la session
        """
        print(f"\n[BruteForce] Lancement de {n_episodes} épisodes aléatoires...")
        tracker = self._run_episodes(n_episodes, phase="BruteForce", log_interval=log_interval, seed=seed)
        tracker.print_summary(label="=== Brute Force — Résultats ===")
        return tracker

    # ------------------------------------------------------------------
    # Replay d'épisodes (affichage console)
    # ------------------------------------------------------------------

    def play_episode(self, seed=None):
        """
        Joue un épisode complet en affichant chaque étape dans la console.
        Utilisé pour le replay aléatoire demandé par le sujet.

        Args:
            seed (int | None): graine pour reset de l'environnement
        """
        render_env = TaxiWrapper(
            reward_mode=self.env.reward_mode,
            render_mode="ansi",
        )
        state, _ = render_env.reset(seed=seed)
        total_reward = 0.0
        step = 0
        terminated = False
        truncated = False

        action_names = ["Sud", "Nord", "Est", "Ouest", "Pickup", "Dropoff"]

        print("\n" + "─" * 40)
        print("  REPLAY — Brute Force")
        print("─" * 40)

        while not (terminated or truncated):
            print(render_env.render())
            action = self.select_action(state)
            state, reward, terminated, truncated, _ = render_env.step(action)
            total_reward += reward
            step += 1
            print(f"  Step {step:3d} | Action: {action_names[action]:<8} | "
                  f"Reward: {reward:+.0f} | Cumul: {total_reward:+.0f}")

        print(render_env.render())
        result = "SUCCÈS" if terminated else "ÉCHEC (timeout)"
        print(f"\n  Résultat : {result} en {step} steps | Reward total : {total_reward:+.0f}")
        print("─" * 40)
        render_env.close()

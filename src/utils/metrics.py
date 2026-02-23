# metrics.py — Collecte, calcul et affichage des métriques de performance

import time
import numpy as np


class EpisodeRecord:
    """Enregistrement des données d'un seul épisode."""

    __slots__ = ("episode", "steps", "total_reward", "success", "duration")

    def __init__(self, episode, steps, total_reward, success, duration):
        self.episode = episode
        self.steps = steps
        self.total_reward = total_reward
        self.success = success          # True si dropoff réussi (terminated=True)
        self.duration = duration        # secondes


class MetricsTracker:
    """
    Collecte les métriques épisode par épisode et calcule les statistiques
    agrégées en fin de session (training ou test).

    Usage typique :
        tracker = MetricsTracker(n_episodes=1000, log_interval=100)
        for ep in range(n_episodes):
            tracker.begin_episode(ep)
            # ... boucle de l'épisode ...
            tracker.end_episode(steps, total_reward, success)
        summary = tracker.summary()
    """

    def __init__(self, n_episodes, log_interval=100, phase="Training"):
        """
        Args:
            n_episodes   (int): nombre total d'épisodes de la session
            log_interval (int): fréquence d'affichage console (tous les N épisodes)
            phase        (str): libellé affiché dans les logs ("Training" | "Test")
        """
        self.n_episodes = n_episodes
        self.log_interval = log_interval
        self.phase = phase

        self.records: list[EpisodeRecord] = []
        self._ep_start_time = None

    # ------------------------------------------------------------------
    # API de collecte (appelée par les agents)
    # ------------------------------------------------------------------

    def begin_episode(self, episode):
        """Démarre le chronomètre pour l'épisode courant."""
        self._ep_start_time = time.perf_counter()

    def end_episode(self, episode, steps, total_reward, success):
        """
        Enregistre les résultats d'un épisode terminé.

        Args:
            episode      (int)  : numéro de l'épisode
            steps        (int)  : nombre de steps effectués
            total_reward (float): reward cumulé sur l'épisode
            success      (bool) : True si le passager a été déposé avec succès
        """
        duration = time.perf_counter() - self._ep_start_time
        self.records.append(EpisodeRecord(episode, steps, total_reward, success, duration))

        if (episode + 1) % self.log_interval == 0:
            self._log(episode)

    # ------------------------------------------------------------------
    # Affichage console périodique
    # ------------------------------------------------------------------

    def _log(self, episode):
        """Affiche les statistiques moyennes sur la dernière fenêtre d'épisodes."""
        window = self.records[-self.log_interval:]
        mean_steps = np.mean([r.steps for r in window])
        mean_reward = np.mean([r.total_reward for r in window])
        success_rate = np.mean([r.success for r in window]) * 100

        print(
            f"[{self.phase}] "
            f"Ep {episode + 1:>{len(str(self.n_episodes))}}/{self.n_episodes} | "
            f"Steps (mean): {mean_steps:6.1f} | "
            f"Reward (mean): {mean_reward:7.2f} | "
            f"Success: {success_rate:5.1f}%"
        )

    # ------------------------------------------------------------------
    # Statistiques agrégées
    # ------------------------------------------------------------------

    def summary(self):
        """
        Calcule et retourne un dict de statistiques agrégées sur tous les épisodes.

        Returns:
            dict avec les clés :
                mean_steps        (float)
                std_steps         (float)
                min_steps         (int)
                max_steps         (int)
                mean_reward       (float)
                std_reward        (float)
                min_reward        (float)
                max_reward        (float)
                success_rate      (float) en pourcentage [0, 100]
                mean_duration     (float) secondes par épisode
                total_duration    (float) secondes au total
                n_episodes        (int)
        """
        if not self.records:
            return {}

        steps = np.array([r.steps for r in self.records])
        rewards = np.array([r.total_reward for r in self.records])
        successes = np.array([r.success for r in self.records])
        durations = np.array([r.duration for r in self.records])

        return {
            "mean_steps": float(np.mean(steps)),
            "std_steps": float(np.std(steps)),
            "min_steps": int(np.min(steps)),
            "max_steps": int(np.max(steps)),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "success_rate": float(np.mean(successes) * 100),
            "mean_duration": float(np.mean(durations)),
            "total_duration": float(np.sum(durations)),
            "n_episodes": len(self.records),
        }

    def print_summary(self, label=None):
        """Affiche un résumé formaté dans la console."""
        s = self.summary()
        if not s:
            print("Aucune donnée enregistrée.")
            return

        title = label or f"=== Résultats — {self.phase} ==="
        sep = "=" * len(title)
        print(f"\n{sep}\n{title}\n{sep}")
        print(f"  Épisodes      : {s['n_episodes']}")
        print(f"  Steps (mean)  : {s['mean_steps']:.1f}  ± {s['std_steps']:.1f}")
        print(f"  Steps (min)   : {s['min_steps']}")
        print(f"  Steps (max)   : {s['max_steps']}")
        print(f"  Reward (mean) : {s['mean_reward']:.2f}  ± {s['std_reward']:.2f}")
        print(f"  Reward (min)  : {s['min_reward']:.2f}")
        print(f"  Reward (max)  : {s['max_reward']:.2f}")
        print(f"  Success rate  : {s['success_rate']:.1f}%")
        print(f"  Time/episode  : {s['mean_duration'] * 1000:.3f} ms")
        print(f"  Total time    : {s['total_duration']:.2f} s")
        print(sep)

    # ------------------------------------------------------------------
    # Accès aux séries temporelles (pour les graphiques)
    # ------------------------------------------------------------------

    def get_series(self):
        """
        Retourne les séries temporelles brutes pour la visualisation.

        Returns:
            dict avec les clés :
                episodes     (np.ndarray int)
                steps        (np.ndarray int)
                rewards      (np.ndarray float)
                successes    (np.ndarray bool)
                durations    (np.ndarray float)
        """
        return {
            "episodes": np.array([r.episode for r in self.records]),
            "steps": np.array([r.steps for r in self.records]),
            "rewards": np.array([r.total_reward for r in self.records]),
            "successes": np.array([r.success for r in self.records]),
            "durations": np.array([r.duration for r in self.records]),
        }

    def rolling_mean(self, key="steps", window=100):
        """
        Calcule la moyenne glissante d'une série pour les courbes d'apprentissage.

        Args:
            key    (str): "steps" | "rewards" | "successes"
            window (int): taille de la fenêtre glissante

        Returns:
            np.ndarray de même longueur que la série (NaN pour les premiers points)
        """
        series = self.get_series()[key].astype(float)
        result = np.full_like(series, np.nan)
        for i in range(len(series)):
            if i + 1 >= window:
                result[i] = np.mean(series[i + 1 - window: i + 1])
        return result

# visualization.py — Génération de tous les graphiques de benchmarking

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

from utils.metrics import MetricsTracker

# Dossier de sortie par défaut
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "plots"
)

# Palette de couleurs cohérente entre tous les graphiques
PALETTE = {
    "BruteForce":           "#e74c3c",
    "Q-Learning (default)": "#e67e22",
    "Q-Learning (tuned)":   "#2ecc71",
    "DQN (default)":        "#3498db",
    "DQN (tuned)":          "#9b59b6",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _save(fig, filename, output_dir):
    """Sauvegarde une figure et l'affiche."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    print(f"[Viz] Graphique sauvegardé → {path}")
    plt.show()
    plt.close(fig)


# ==================================================================
# Graphique 1 — Courbes d'apprentissage (reward moyen glissant)
# ==================================================================

def plot_learning_curves(
    trackers: dict[str, MetricsTracker],
    window: int = 100,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    """
    Courbes de reward cumulé par épisode avec moyenne glissante.
    Permet de comparer la vitesse de convergence de chaque algorithme.

    Args:
        trackers   (dict): {label: MetricsTracker}
        window     (int) : taille de la fenêtre de lissage
        output_dir (str) : dossier de sauvegarde
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, tracker in trackers.items():
        series = tracker.get_series()
        episodes = series["episodes"]
        rewards = series["rewards"]
        smoothed = tracker.rolling_mean(key="rewards", window=window)
        color = PALETTE.get(label, None)

        ax.plot(episodes, rewards, alpha=0.15, color=color, linewidth=0.8)
        ax.plot(episodes, smoothed, label=f"{label} (moy. {window} ep)",
                color=color, linewidth=2)

    ax.set_title("Courbes d'apprentissage — Reward cumulé par épisode", fontsize=13)
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Reward cumulé")
    ax.legend(loc="lower right")
    _save(fig, "01_learning_curves.png", output_dir)


# ==================================================================
# Graphique 2 — Évolution du nombre de steps
# ==================================================================

def plot_steps_evolution(
    trackers: dict[str, MetricsTracker],
    window: int = 100,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    """
    Évolution du nombre de steps par épisode avec moyenne glissante.
    Visualise la vitesse de convergence vers une politique efficace.

    Args:
        trackers   (dict): {label: MetricsTracker}
        window     (int) : taille de la fenêtre de lissage
        output_dir (str) : dossier de sauvegarde
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, tracker in trackers.items():
        series = tracker.get_series()
        episodes = series["episodes"]
        smoothed = tracker.rolling_mean(key="steps", window=window)
        color = PALETTE.get(label, None)

        ax.plot(episodes, smoothed, label=f"{label} (moy. {window} ep)",
                color=color, linewidth=2)

    ax.set_title("Évolution du nombre de steps par épisode", fontsize=13)
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Steps (moyenne glissante)")
    ax.legend(loc="upper right")
    _save(fig, "02_steps_evolution.png", output_dir)


# ==================================================================
# Graphique 3 — Évolution d'epsilon
# ==================================================================

def plot_epsilon_decay(
    epsilon_histories: dict[str, list[float]],
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    """
    Courbe de décroissance d'epsilon pour les agents RL.
    Visualise la transition exploration → exploitation.

    Args:
        epsilon_histories (dict): {label: liste des valeurs epsilon par épisode}
        output_dir        (str) : dossier de sauvegarde
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    for label, history in epsilon_histories.items():
        color = PALETTE.get(label, None)
        ax.plot(history, label=label, color=color, linewidth=2)

    ax.set_title("Décroissance d'epsilon (exploration → exploitation)", fontsize=13)
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Valeur d'ε")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend()
    _save(fig, "03_epsilon_decay.png", output_dir)


# ==================================================================
# Graphique 4 — Tableau comparatif final
# ==================================================================

def plot_summary_table(
    summaries: dict[str, dict],
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    """
    Génère un tableau comparatif des métriques agrégées de tous les algorithmes.

    Args:
        summaries  (dict): {label: summary_dict} (depuis MetricsTracker.summary())
        output_dir (str) : dossier de sauvegarde
    """
    rows = []
    for label, s in summaries.items():
        rows.append({
            "Algorithme": label,
            "Mean Steps": f"{s['mean_steps']:.1f}",
            "Std Steps": f"± {s['std_steps']:.1f}",
            "Mean Reward": f"{s['mean_reward']:.2f}",
            "Success Rate": f"{s['success_rate']:.1f}%",
            "Time/ep (ms)": f"{s['mean_duration'] * 1000:.2f}",
        })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, len(rows) * 0.6 + 1.2))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Colorer l'en-tête
    for j in range(len(df.columns)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alterner les couleurs des lignes
    for i in range(1, len(rows) + 1):
        for j in range(len(df.columns)):
            table[i, j].set_facecolor("#ecf0f1" if i % 2 == 0 else "white")

    ax.set_title("Tableau comparatif — Résultats de test", fontsize=13, pad=20)
    _save(fig, "04_summary_table.png", output_dir)
    print("\n" + df.to_string(index=False))


# ==================================================================
# Graphique 5 — Distribution des steps (boxplot)
# ==================================================================

def plot_steps_distribution(
    trackers: dict[str, MetricsTracker],
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    """
    Boxplot comparatif de la distribution des steps sur les épisodes de test.
    Visualise la dispersion et la médiane de chaque algorithme.

    Args:
        trackers   (dict): {label: MetricsTracker}
        output_dir (str) : dossier de sauvegarde
    """
    data = []
    for label, tracker in trackers.items():
        series = tracker.get_series()
        for s in series["steps"]:
            data.append({"Algorithme": label, "Steps": int(s)})

    df = pd.DataFrame(data)
    order = list(trackers.keys())
    colors = [PALETTE.get(l, "#95a5a6") for l in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=df, x="Algorithme", y="Steps",
        order=order, palette=colors, ax=ax,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
    )
    ax.set_title("Distribution du nombre de steps — épisodes de test", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Steps par épisode")
    plt.xticks(rotation=15)
    _save(fig, "05_steps_distribution.png", output_dir)


# ==================================================================
# Graphique 6 — Heatmap sensibilité aux hyperparamètres
# ==================================================================

def plot_hyperparam_heatmap(
    results: dict,
    param_x: str,
    param_y: str,
    metric: str = "mean_steps",
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    """
    Heatmap de la sensibilité d'une métrique à deux hyperparamètres.
    Permet d'identifier les meilleures combinaisons de paramètres.

    Args:
        results  (dict): {(val_x, val_y): summary_dict}
                         ex: {(0.1, 0.99): {"mean_steps": 15.2, ...}, ...}
        param_x  (str) : nom du paramètre en colonne (ex: "alpha")
        param_y  (str) : nom du paramètre en ligne   (ex: "gamma")
        metric   (str) : métrique à afficher ("mean_steps" | "mean_reward" | "success_rate")
        output_dir(str): dossier de sauvegarde
    """
    xs = sorted(set(k[0] for k in results))
    ys = sorted(set(k[1] for k in results), reverse=True)

    matrix = np.array([
        [results[(x, y)][metric] for x in xs]
        for y in ys
    ])

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = "RdYlGn_r" if metric == "mean_steps" else "RdYlGn"
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        xticklabels=xs,
        yticklabels=ys,
        cmap=cmap,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"Sensibilité aux hyperparamètres — {metric}", fontsize=13)
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    _save(fig, f"06_heatmap_{param_x}_{param_y}.png", output_dir)


# ==================================================================
# Graphique 7 — Comparaison reward shaping
# ==================================================================

def plot_reward_shaping_comparison(
    trackers: dict[str, MetricsTracker],
    window: int = 100,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    """
    Compare l'impact des différentes stratégies de reward shaping
    sur la courbe d'apprentissage (steps par épisode).

    Args:
        trackers   (dict): {label_shaping: MetricsTracker}
        window     (int) : taille de la fenêtre de lissage
        output_dir (str) : dossier de sauvegarde
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, tracker in trackers.items():
        series = tracker.get_series()
        episodes = series["episodes"]
        smoothed_steps   = tracker.rolling_mean(key="steps",   window=window)
        smoothed_rewards = tracker.rolling_mean(key="rewards", window=window)

        axes[0].plot(episodes, smoothed_steps,   label=label, linewidth=2)
        axes[1].plot(episodes, smoothed_rewards, label=label, linewidth=2)

    axes[0].set_title("Steps par épisode selon le reward shaping")
    axes[0].set_xlabel("Épisode")
    axes[0].set_ylabel("Steps (moy. glissante)")
    axes[0].legend()

    axes[1].set_title("Reward cumulé selon le reward shaping")
    axes[1].set_xlabel("Épisode")
    axes[1].set_ylabel("Reward (moy. glissante)")
    axes[1].legend()

    fig.suptitle("Impact du Reward Shaping sur l'apprentissage", fontsize=14, y=1.02)
    _save(fig, "07_reward_shaping.png", output_dir)

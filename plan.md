# Plan de Projet - Taxi Driver (T-AIA-902)
## Reinforcement Learning — Plan d'implémentation de A à Z

---

## Vue d'ensemble

**Objectif** : Entraîner un agent RL à résoudre l'environnement `Taxi-v3` de Gymnasium (ex-OpenAI Gym). L'agent (un taxi) doit récupérer un passager à une position aléatoire et le déposer à l'une des 4 destinations possibles, en un minimum d'étapes.

**Algorithme principal recommandé** : Q-Learning (baseline) + Deep Q-Learning (optimisé)

**Livrables** :
- Code source complet sur GitHub
- Un rapport de benchmarking (document)
- Le programme avec deux modes (user mode / time-limited mode)

---

## Phase 0 — Mise en place du projet

### 0.1 — Initialisation du dépôt Git

- Créer le dépôt GitHub nommé `TaxiDriver`
- Initialiser le dépôt localement avec `git init`
- Créer un `.gitignore` adapté au Python (exclure : `__pycache__/`, `*.pyc`, `*.pth`, `.env`, `venv/`, `results/models/*.pkl` si trop lourds, les fichiers binaires temporaires)
- Créer un `README.md` minimal (nom du projet, description, instructions de lancement)

### 0.2 — Structure du projet

Définir l'arborescence suivante **avant** d'écrire la moindre ligne de code :

```
TaxiDriver/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── brute_force.py         # Algorithme naïf aléatoire
│   │   ├── q_learning.py          # Q-Learning tabulaire (baseline RL)
│   │   └── dqn.py                 # Deep Q-Network (algorithme optimisé principal)
│   ├── environment/
│   │   ├── __init__.py
│   │   └── taxi_wrapper.py        # Wrapper Gymnasium pour centraliser les interactions
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Collecte et agrégation des métriques
│   │   ├── visualization.py       # Génération de tous les graphiques
│   │   └── replay_buffer.py       # Experience Replay Buffer pour DQN
│   └── main.py                    # Point d'entrée du programme (modes user/time-limited)
├── results/
│   ├── models/                    # Sauvegarde des modèles entraînés (.pkl, .pt)
│   └── plots/                     # Graphiques générés automatiquement
├── report/
│   └── report.md                  # Rapport de benchmarking (ou .pdf exporté)
├── requirements.txt
├── plan.md                        # Ce fichier
└── README.md
```

### 0.3 — Environnement Python et dépendances

- Créer un environnement virtuel Python 3.10+ (`venv` ou `conda`)
- Identifier les dépendances nécessaires et les inscrire dans `requirements.txt` :
  - `gymnasium` — environnement Taxi-v3
  - `numpy` — calculs matriciels (Q-table)
  - `torch` — réseau de neurones pour DQN
  - `matplotlib` — graphiques de performance
  - `pandas` — tableaux de résultats
  - `tqdm` — barres de progression pour l'entraînement
  - `seaborn` — visualisations avancées (optionnel)
  - `pickle` / `json` — sauvegarde des modèles et des résultats

---

## Phase 1 — Compréhension approfondie de l'environnement Taxi-v3

### 1.1 — Étudier l'environnement Taxi-v3

Avant tout code, comprendre exactement l'environnement :

- **Grille** : 5×5 = 25 positions possibles pour le taxi
- **Passager** : peut être à 4 positions désignées (R, G, Y, B) ou dans le taxi → 5 positions
- **Destination** : 4 positions désignées (R, G, Y, B)
- **Espace des états** : 25 × 5 × 4 = **500 états discrets**
- **Espace des actions** : 6 actions discrètes
  - 0 : aller au Sud
  - 1 : aller au Nord
  - 2 : aller à l'Est
  - 3 : aller à l'Ouest
  - 4 : prendre le passager (PICKUP)
  - 5 : déposer le passager (DROPOFF)
- **Système de récompenses (rewards) par défaut** :
  - `-1` par étape (pénalité de déplacement)
  - `+20` pour un dépôt réussi
  - `-10` pour un pickup ou dropoff illégal (mauvais endroit)
- **Terminaison** : l'épisode se termine quand le passager est déposé à la bonne destination, ou après 200 étapes max
- **API Gymnasium** à maîtriser :
  - `env.reset()` → retourne l'état initial
  - `env.step(action)` → retourne `(next_state, reward, terminated, truncated, info)`
  - `env.render()` → affiche l'environnement en ASCII

### 1.2 — Définir les métriques de performance

Avant de coder quoi que ce soit, définir précisément ce qu'on va mesurer :

- **Steps per episode** : nombre d'actions avant la fin de l'épisode
- **Total reward per episode** : somme des récompenses sur tout l'épisode
- **Mean steps (test)** : moyenne des steps sur les épisodes de test
- **Mean reward (test)** : moyenne des récompenses sur les épisodes de test
- **Success rate** : pourcentage d'épisodes terminés avec succès (dropoff réussi)
- **Training time** : durée totale d'entraînement
- **Time per episode** : durée moyenne par épisode

---

## Phase 2 — Algorithme Brute Force (point de comparaison naïf)

### 2.1 — Conception de l'algorithme

L'algorithme brute force est un agent **aléatoire pur** : à chaque étape, il choisit une action uniformément au hasard parmi les 6 actions disponibles, sans aucun apprentissage.

- Pas de politique apprise
- Pas de mémoire entre les épisodes
- Sert de **borne inférieure** de performance

Référence attendue : environ **350 steps** en moyenne pour finir un épisode (s'il y arrive).

### 2.2 — Ce que le module `brute_force.py` doit faire

- Implémenter une classe `BruteForceAgent` avec :
  - `select_action(state)` → retourne une action aléatoire
  - `train(env, n_episodes)` → boucle d'entraînement (ici aucun apprentissage, juste exécution)
  - `test(env, n_episodes)` → boucle de test avec collecte des métriques
- Collecter et retourner à chaque épisode : steps, reward cumulé, succès ou non, durée

---

## Phase 3 — Algorithme Q-Learning Tabulaire (baseline RL non-optimisé)

### 3.1 — Principe du Q-Learning

Le Q-Learning est un algorithme off-policy model-free qui maintient une **Q-table** de taille `(n_states × n_actions)` = `(500 × 6)` pour Taxi-v3.

À chaque étape, la Q-table est mise à jour par la règle de Bellman :
```
Q(s, a) ← Q(s, a) + α × [r + γ × max Q(s', a') - Q(s, a)]
```

**Paramètres hyperparamètres** :
- `alpha` (α) : taux d'apprentissage (learning rate) — contrôle la vitesse de mise à jour
- `gamma` (γ) : facteur de discount — pondère l'importance des récompenses futures
- `epsilon` (ε) : taux d'exploration initial (epsilon-greedy)
- `epsilon_decay` : décroissance de ε à chaque épisode (exploration → exploitation)
- `epsilon_min` : valeur minimale de ε
- `n_training_episodes` : nombre d'épisodes d'entraînement
- `n_testing_episodes` : nombre d'épisodes de test

### 3.2 — Stratégie epsilon-greedy

- Pendant l'entraînement : avec probabilité ε, choisir une action aléatoire (exploration) ; sinon, choisir `argmax Q(s, .)` (exploitation)
- ε diminue progressivement à chaque épisode via `epsilon_decay`
- Pendant le test : ε = 0 (pure exploitation, politique greedy)

### 3.3 — Ce que le module `q_learning.py` doit faire

- Implémenter une classe `QLearningAgent` avec :
  - `__init__(alpha, gamma, epsilon, epsilon_decay, epsilon_min, n_states, n_actions)` → initialise la Q-table à zéro
  - `select_action(state, training=True)` → epsilon-greedy si training, greedy sinon
  - `update(state, action, reward, next_state, done)` → applique la règle de Bellman
  - `train(env, n_episodes)` → boucle d'entraînement complète, retourne les métriques
  - `test(env, n_episodes)` → boucle de test avec métriques
  - `save(path)` → sauvegarde la Q-table sur disque (pickle ou numpy)
  - `load(path)` → charge une Q-table existante

### 3.4 — Version non-optimisée (premier baseline RL)

- Lancer d'abord avec des hyperparamètres **par défaut, non-tuned** (ex: α=0.1, γ=0.99, ε=1.0, decay=0.995)
- Enregistrer les résultats de cette version comme **deuxième point de comparaison**
- C'est à partir de cette baseline qu'on va itérer pour améliorer

---

## Phase 4 — Algorithme Deep Q-Learning / DQN (algorithme principal optimisé)

### 4.1 — Pourquoi DQN pour Taxi-v3 ?

Taxi-v3 est un environnement discret avec seulement 500 états, donc le Q-Learning tabulaire suffit techniquement. Cependant, le sujet recommande DQN ou Monte Carlo. On implémente DQN pour :
- Démontrer la maîtrise des algorithmes deep RL
- Avoir un algorithme scalable à des environnements plus complexes (extension possible)
- Comparer DQN vs Q-Learning tabulaire sur le même problème

> **Note** : Si DQN n'apporte pas de gain sur Taxi-v3, cela sera commenté et justifié dans le rapport — c'est une observation valide et attendue.

### 4.2 — Architecture du réseau neuronal (DQN)

- **Input** : l'état encodé en one-hot vector (taille 500) ou directement l'entier encodé
- **Couches cachées** : 2 couches fully-connected (ex: 128 → 64 neurones, activation ReLU)
- **Output** : Q-values pour chaque action (taille 6)
- **Loss** : Mean Squared Error (MSE) ou Huber Loss entre Q-value prédit et cible

### 4.3 — Composants spécifiques au DQN

**Experience Replay Buffer** (`replay_buffer.py`) :
- Stocke les transitions `(state, action, reward, next_state, done)` dans un buffer circulaire
- Capacité maximale : ex. 10 000 transitions
- À chaque étape, tire un mini-batch aléatoire de taille (ex. 64) pour la mise à jour
- Brise la corrélation temporelle entre les transitions (crucial pour la stabilité)

**Target Network** :
- Réseau neuronal identique au réseau principal, mais ses poids sont mis à jour moins fréquemment
- Tous les N steps (ex. N=100 ou N=500), copier les poids du réseau principal vers le target network
- Stabilise l'entraînement en évitant les "moving targets"

**Optimiseur** :
- Adam ou RMSprop avec un learning rate réduit (ex. 0.001 ou 0.0005)

### 4.4 — Ce que le module `dqn.py` doit faire

- Implémenter une classe `DQNAgent` avec :
  - `__init__(state_size, action_size, lr, gamma, epsilon, epsilon_decay, epsilon_min, batch_size, buffer_capacity, target_update_freq)` → initialise réseau, target network, buffer
  - `select_action(state, training=True)` → epsilon-greedy
  - `store_transition(state, action, reward, next_state, done)` → stockage dans le buffer
  - `learn()` → tire un batch, calcule les cibles via target network, rétropropage la loss
  - `update_target_network()` → copie les poids vers le target network
  - `train(env, n_episodes)` → boucle d'entraînement complète
  - `test(env, n_episodes)` → boucle de test avec métriques
  - `save(path)` → sauvegarde les poids du modèle (torch.save)
  - `load(path)` → charge les poids sauvegardés

### 4.5 — Stratégie de tuning des hyperparamètres

Explorer systématiquement les hyperparamètres suivants (pour le rapport de benchmarking) :
- `alpha` / `lr` : [0.01, 0.001, 0.0005, 0.0001]
- `gamma` : [0.90, 0.95, 0.99, 0.999]
- `epsilon_decay` : [0.99, 0.995, 0.999]
- `batch_size` : [32, 64, 128]
- `target_update_freq` : [50, 100, 500]
- Nombre d'épisodes d'entraînement : [1000, 5000, 10000]

Pour chaque combinaison testée, enregistrer : mean steps (test), mean reward (test), training time.

---

## Phase 5 — Reward Shaping (optimisation des récompenses)

### 5.1 — Pourquoi modifier les récompenses ?

Les récompenses par défaut de Taxi-v3 sont sparses (rares). On peut guider l'apprentissage en ajoutant des récompenses intermédiaires via un wrapper.

### 5.2 — Stratégies de reward shaping à tester

- **Récompense de distance** : récompense proportionnelle à la réduction de distance Manhattan entre le taxi et le passager (phase pickup), puis entre le taxi et la destination (phase dropoff)
- **Pénalité de temps plus forte** : augmenter la pénalité par step (ex. -2 au lieu de -1) pour forcer l'efficacité
- **Bonus de succès** : augmenter le bonus de dropoff réussi (ex. +50 au lieu de +20)
- **Version baseline** : garder les récompenses d'origine pour comparaison

Chaque configuration de reward shaping sera testée et ses résultats inclus dans le rapport.

---

## Phase 6 — Module Environnement (`taxi_wrapper.py`)

### 6.1 — Rôle du wrapper

Ce module centralise toutes les interactions avec Gymnasium pour éviter la duplication et faciliter le reward shaping.

### 6.2 — Ce que le wrapper doit faire

- Initialiser l'environnement Gymnasium `Taxi-v3`
- Exposer les attributs essentiels : `n_states`, `n_actions`, `state_space`, `action_space`
- Méthode `reset()` → wrapper de `env.reset()`
- Méthode `step(action)` → wrapper de `env.step(action)`, avec possibilité d'injecter des récompenses shapées
- Méthode `render()` → affiche l'état courant en ASCII
- Méthode `get_state_features(state)` → encode l'état en one-hot pour DQN si nécessaire
- Méthode `decode_state(state)` → décompose l'entier d'état en (row, col, passenger_loc, destination)

---

## Phase 7 — Module Métriques (`metrics.py`)

### 7.1 — Ce que le module doit collecter

À chaque épisode (training et test), enregistrer dans des structures de données :
- Numéro de l'épisode
- Nombre de steps
- Reward total cumulé
- Succès ou non (dropoff réussi = `terminated` sans `truncated`)
- Durée de l'épisode (en secondes)

### 7.2 — Ce que le module doit calculer

À la fin d'une session (training ou test), calculer et afficher :
- **Mean steps** (moyenne du nombre de steps par épisode)
- **Mean reward** (moyenne du reward cumulé par épisode)
- **Success rate** (% d'épisodes réussis)
- **Best episode** (min steps, max reward)
- **Worst episode** (max steps, min reward)
- **Std deviation** des steps et des rewards
- **Training time total** et **mean time per episode**

### 7.3 — Affichage console pendant l'entraînement

Afficher régulièrement (ex. tous les 100 épisodes) :
```
Episode [900/1000] | Mean Steps (last 100): 18.3 | Mean Reward: 7.2 | ε: 0.081
```

---

## Phase 8 — Module Visualisation (`visualization.py`)

### 8.1 — Graphiques à générer

Tous les graphiques doivent être sauvegardés en PNG dans `results/plots/` et affichés à la demande.

**Graphique 1 — Courbe d'apprentissage (Training Reward)** :
- X : numéro d'épisode
- Y : reward cumulé par épisode (avec moyenne glissante sur 100 épisodes)
- Une courbe par algorithme sur le même graphique (Brute Force, Q-Learning, DQN)

**Graphique 2 — Évolution du nombre de steps** :
- X : numéro d'épisode
- Y : steps par épisode (avec moyenne glissante)
- Comparer la vitesse de convergence des algorithmes

**Graphique 3 — Évolution d'epsilon** :
- X : numéro d'épisode
- Y : valeur de ε
- Visualiser la transition exploration → exploitation

**Graphique 4 — Tableau comparatif final** :
- Tableau (pandas DataFrame exporté en image) avec les colonnes :
  `Algorithme | Mean Steps | Mean Reward | Success Rate | Training Time`
- Une ligne par algorithme et par configuration testée

**Graphique 5 — Distribution des steps (boxplot ou histogramme)** :
- Comparer la dispersion des performances entre les algorithmes pendant le test

**Graphique 6 — Sensibilité aux hyperparamètres (heatmap)** :
- Ex. heatmap `alpha` × `gamma` avec la couleur = mean steps (test)
- Permet de visualiser les meilleures combinaisons de paramètres

**Graphique 7 — Reward shaping comparison** :
- Comparer les différentes configurations de récompenses sur les mêmes métriques

### 8.2 — Affichage d'épisodes aléatoires

- Après le test, sélectionner aléatoirement N épisodes (ex. 3) et les rejouer en affichant `env.render()` étape par étape dans la console
- Afficher l'état, l'action choisie, la récompense et le cumul à chaque step

---

## Phase 9 — Point d'entrée principal (`main.py`)

### 9.1 — Flux général du programme

Le programme doit :
1. Afficher un menu de sélection du mode
2. Demander le nombre d'épisodes d'entraînement ET de test
3. Lancer l'entraînement
4. Lancer le test
5. Afficher les métriques
6. Afficher des épisodes aléatoires
7. Générer et sauvegarder les graphiques

### 9.2 — Mode 1 : User Mode (mode interactif)

**Description** : L'utilisateur contrôle tous les paramètres de l'algorithme.

À l'entrée du mode, demander à l'utilisateur :
- Choix de l'algorithme : `[1] Brute Force | [2] Q-Learning | [3] DQN`
- Nombre d'épisodes d'entraînement
- Nombre d'épisodes de test
- **Si Q-Learning** : `alpha`, `gamma`, `epsilon`, `epsilon_decay`, `epsilon_min`
- **Si DQN** : `lr`, `gamma`, `epsilon`, `epsilon_decay`, `batch_size`, `target_update_freq`
- Choix du reward shaping : `[1] Défaut | [2] Distance-based | [3] Custom`

Valider les entrées utilisateur (plages acceptables, types corrects).

### 9.3 — Mode 2 : Time-Limited Mode

**Description** : L'utilisateur spécifie une durée maximale d'entraînement. Les paramètres sont pré-réglés aux valeurs optimales trouvées pendant le benchmarking.

À l'entrée du mode, demander à l'utilisateur :
- Durée maximale d'entraînement (en secondes)
- Nombre d'épisodes de test

L'entraînement tourne en boucle avec les paramètres optimaux et s'arrête dès que le temps est écoulé.

Afficher à la fin : les métriques de l'agent entraîné dans ce temps imparti.

### 9.4 — Interface console

Exemple de flow d'entrée/sortie attendu :
```
=== TAXI DRIVER - Reinforcement Learning ===
Mode: [1] User  [2] Time-Limited
> 1

Algorithm: [1] Brute Force  [2] Q-Learning  [3] DQN
> 3

Training episodes: 5000
Testing episodes: 100

Learning rate [default: 0.001]: 0.001
Gamma [default: 0.99]: 0.99
Epsilon [default: 1.0]: 1.0
...

Training DQN... 100%|████████| 5000/5000 [01:23<00:00]

=== TEST RESULTS ===
Mean Steps    : 12.4
Mean Reward   : 7.8
Success Rate  : 98%
Mean Time/Ep  : 0.003s

Displaying 3 random episodes...
```

---

## Phase 10 — Benchmarking complet

### 10.1 — Protocole de benchmark

Pour chaque algorithme et configuration, effectuer **les mêmes 100 épisodes de test** (avec la même seed aléatoire si possible) pour garantir la comparabilité.

**Run 1 — Brute Force** :
- 0 épisodes d'entraînement, 100 épisodes de test
- Collecter : mean steps, mean reward, success rate, mean time

**Run 2 — Q-Learning non-optimisé** :
- Paramètres par défaut (α=0.1, γ=0.99, ε=1.0, decay=0.995)
- 1000 épisodes d'entraînement, 100 épisodes de test
- Collecter toutes les métriques

**Run 3 — Q-Learning optimisé** :
- Meilleurs hyperparamètres trouvés lors du tuning
- 10 000 épisodes d'entraînement, 100 épisodes de test

**Run 4 — DQN non-optimisé** :
- Paramètres par défaut
- 5000 épisodes d'entraînement, 100 épisodes de test

**Run 5 — DQN optimisé** :
- Meilleurs hyperparamètres + best reward shaping
- 10 000 épisodes d'entraînement, 100 épisodes de test

**Run 6 (optionnel) — Monte Carlo** :
- Implémenter un agent Monte Carlo ES ou first-visit MC pour diversifier les comparaisons

### 10.2 — Métriques à comparer dans le tableau final

| Algorithme | Train Episodes | Mean Steps | Mean Reward | Success Rate | Training Time |
|---|---|---|---|---|---|
| Brute Force | 0 | ~350 | très négatif | ~0% | 0s |
| Q-Learning (default) | 1000 | ~X | ... | ... | ... |
| Q-Learning (tuned) | 10000 | ~X | ... | ... | ... |
| DQN (default) | 5000 | ~X | ... | ... | ... |
| DQN (tuned) | 10000 | ~20 | +7 à +8 | ~98% | ... |

---

## Phase 11 — Rapport de Benchmarking

### 11.1 — Structure du rapport

Le rapport (`report/report.md`) doit contenir les sections suivantes :

**1. Introduction**
- Présentation du problème Taxi-v3
- Description de l'espace d'états, des actions et des récompenses
- Objectif : minimiser le nombre de steps

**2. Algorithme Brute Force**
- Description
- Résultats : mean steps, mean reward, success rate
- Justification de son utilité comme borne inférieure

**3. Q-Learning Tabulaire**
- Explication mathématique de la règle de Bellman
- Architecture : Q-table 500×6
- Résultats non-optimisés vs optimisés
- Analyse de l'impact de chaque hyperparamètre (avec graphiques)
- Justification des hyperparamètres finaux choisis

**4. Deep Q-Learning (DQN)**
- Explication de l'architecture réseau
- Description de l'Experience Replay et du Target Network
- Résultats non-optimisés vs optimisés
- Comparaison avec Q-Learning tabulaire (avantages et limites sur Taxi-v3)

**5. Reward Shaping**
- Description des différentes configurations testées
- Tableau comparatif des résultats
- Analyse : quelle configuration de reward permet la convergence la plus rapide ?

**6. Stratégie d'optimisation**
- Processus de tuning des hyperparamètres (grille de recherche)
- Heatmaps des combinaisons testées
- Justification des paramètres finaux retenus

**7. Conclusion**
- Comparaison finale de tous les algorithmes
- Recommandation de l'algorithme optimal pour Taxi-v3
- Limites et perspectives (extension 2 passagers)

**8. Annexes**
- Tous les graphiques
- Tableaux de données brutes

### 11.2 — Justifications attendues dans le rapport

- Pourquoi epsilon-greedy plutôt que softmax ou UCB ?
- Pourquoi ce taux de décroissance de ε ?
- Pourquoi ce gamma ? (valeur proche de 1 = plus de valeur aux récompenses futures)
- Pourquoi l'Experience Replay aide-t-il ? (brise la corrélation temporelle)
- Pourquoi un Target Network ? (stabilise les cibles Q pendant l'apprentissage)
- Différence entre on-policy (SARSA) et off-policy (Q-Learning/DQN) et pourquoi on a choisi off-policy

---

## Phase 12 — Extension optionnelle (2 passagers)

> À implémenter uniquement si tout le reste est terminé et validé.

### 12.1 — Définition du problème étendu

- 2 passagers à récupérer simultanément (ou séquentiellement)
- 4 destinations possibles pour chaque passager
- L'objectif est d'optimiser la route totale (minimiser la distance parcourue)

### 12.2 — Problèmes soulevés par l'extension

- L'espace d'états explose : Q-Learning tabulaire devient impraticable → DQN indispensable
- Le problème avec 2 passagers ressemble au **Travelling Salesman Problem (TSP)** → NP-hard
- Il faudra concevoir un environnement custom avec Gymnasium (hériter de `gym.Env`)

### 12.3 — Plan de l'extension

- Créer `src/environment/taxi_extended.py` : environnement Gymnasium custom
  - Définir le nouvel espace d'états (position taxi + positions des 2 passagers + 2 destinations + status de chaque passager)
  - Définir les nouvelles récompenses
- Adapter `dqn.py` pour le nouvel espace d'états (plus grand)
- Ajouter un reward shaping basé sur la distance totale optimale (heuristique)
- Comparer avec l'environnement original

---

## Phase 13 — Finalisation et livraison

### 13.1 — Tests de non-régression

- Vérifier que le programme se lance sans erreur dans les deux modes
- Tester avec des valeurs extrêmes de paramètres (epsilon=0, gamma=0, etc.)
- Vérifier que la sauvegarde et le chargement de modèles fonctionnent
- Vérifier que tous les graphiques se génèrent correctement
- Vérifier que les épisodes aléatoires s'affichent bien en console

### 13.2 — Nettoyage du code

- Supprimer tout fichier temporaire, binaire, ou de cache (`__pycache__`, `.pyc`, etc.)
- Vérifier que le `.gitignore` exclut bien tous ces fichiers
- S'assurer que le `README.md` contient des instructions d'installation et de lancement claires :
  ```
  pip install -r requirements.txt
  python src/main.py
  ```

### 13.3 — Vérification du `.gitignore`

S'assurer que les fichiers suivants sont exclus du dépôt :
- `__pycache__/`, `*.pyc`, `*.pyo`
- `venv/`, `.env`
- Gros fichiers de modèles si besoin (ou les inclure avec une note)
- Fichiers IDE (`.vscode/`, `.idea/`)

### 13.4 — Push final

- Commit final avec tous les fichiers sources, le rapport, le `requirements.txt`, et le `README.md`
- Vérifier que le dépôt est public et accessible
- Vérifier le nom du dépôt : `TaxiDriver`

---

## Récapitulatif des fichiers à créer

| Fichier | Rôle |
|---|---|
| `src/main.py` | Point d'entrée, gestion des modes user/time-limited |
| `src/agents/brute_force.py` | Agent aléatoire (baseline naïf) |
| `src/agents/q_learning.py` | Agent Q-Learning tabulaire (baseline RL) |
| `src/agents/dqn.py` | Agent Deep Q-Network (algorithme principal) |
| `src/environment/taxi_wrapper.py` | Wrapper Gymnasium + reward shaping |
| `src/utils/metrics.py` | Collecte, calcul et affichage des métriques |
| `src/utils/visualization.py` | Génération des graphiques matplotlib |
| `src/utils/replay_buffer.py` | Experience Replay Buffer pour DQN |
| `requirements.txt` | Dépendances Python |
| `report/report.md` | Rapport de benchmarking complet |
| `README.md` | Instructions d'installation et de lancement |
| `.gitignore` | Exclusion des fichiers inutiles |

---

## Ordre d'implémentation recommandé

1. Setup du projet (Phase 0) — structure, dépendances, Git
2. Wrapper environnement (Phase 6) — base de tout le reste
3. Module métriques (Phase 7) — nécessaire pour mesurer dès le début
4. Brute Force (Phase 2) — valider que l'environnement fonctionne
5. Q-Learning non-optimisé (Phase 3) — premier benchmark RL
6. Module visualisation (Phase 8) — commencer à tracer les courbes
7. Q-Learning optimisé — tuning des hyperparamètres
8. DQN (Phase 4) — algorithme principal
9. Reward shaping (Phase 5) — optimisation des récompenses
10. Main.py avec les deux modes (Phase 9) — interface utilisateur
11. Benchmarking complet (Phase 10) — tous les runs comparatifs
12. Rapport (Phase 11) — rédaction du document final
13. Finalisation et livraison (Phase 13) — nettoyage et push

---

*Document généré le 23/02/2026 — T-AIA-902 Taxi Driver*

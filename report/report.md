# Rapport de Benchmarking — Taxi Driver
## T-AIA-902 — Reinforcement Learning

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Environnement Taxi-v3](#2-environnement-taxi-v3)
3. [Algorithme Brute Force](#3-algorithme-brute-force)
4. [Q-Learning Tabulaire](#4-q-learning-tabulaire)
5. [Deep Q-Network (DQN)](#5-deep-q-network-dqn)
6. [Reward Shaping](#6-reward-shaping)
7. [Stratégie d'optimisation des hyperparamètres](#7-stratégie-doptimisation-des-hyperparamètres)
8. [Tableau comparatif final](#8-tableau-comparatif-final)
9. [Conclusion](#9-conclusion)

---

## 1. Introduction

Ce rapport présente les résultats obtenus sur le problème **Taxi-v3** de la bibliothèque Gymnasium dans le cadre du projet T-AIA-902. L'objectif est d'entraîner un agent de Reinforcement Learning capable de récupérer un passager à une position aléatoire sur une grille 5×5 et de le déposer à la destination correcte, en un minimum d'étapes.

Trois familles d'algorithmes ont été implémentées et comparées :

- **Brute Force** — agent aléatoire, aucun apprentissage, sert de borne inférieure
- **Q-Learning tabulaire** — algorithme off-policy model-free classique, baseline RL
- **Deep Q-Network (DQN)** — approximation des Q-valeurs par réseau de neurones, algorithme principal

Pour chaque algorithme, deux versions sont évaluées : une version avec des paramètres **par défaut** (non-optimisés) et une version **optimisée** après tuning. Cette progression permet de mesurer l'apport réel de chaque optimisation.

---

## 2. Environnement Taxi-v3

### 2.1 Description

L'environnement Taxi-v3 est une grille de 5×5 cases. Quatre emplacements désignés sont présents sur la grille, identifiés par les lettres R (Red), G (Green), Y (Yellow) et B (Blue) :

```
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
```

À chaque épisode :
- Le taxi est placé à une position aléatoire
- Le passager est placé aléatoirement dans l'un des 4 emplacements
- La destination est choisie aléatoirement parmi les 3 emplacements restants

L'épisode se termine lorsque le passager est déposé à la bonne destination, ou après 200 steps (truncation).

### 2.2 Espace des états

L'espace d'états est discret et de taille **500** :

| Composante | Valeurs | Taille |
|---|---|---|
| Position du taxi (ligne) | 0 à 4 | 5 |
| Position du taxi (colonne) | 0 à 4 | 5 |
| Position du passager | 0=R, 1=G, 2=Y, 3=B, 4=dans le taxi | 5 |
| Destination | 0=R, 1=G, 2=Y, 3=B | 4 |
| **Total** | 5 × 5 × 5 × 4 | **500** |

### 2.3 Espace des actions

6 actions discrètes :

| ID | Action | Description |
|---|---|---|
| 0 | Sud | Déplacer le taxi vers le bas |
| 1 | Nord | Déplacer le taxi vers le haut |
| 2 | Est | Déplacer le taxi vers la droite |
| 3 | Ouest | Déplacer le taxi vers la gauche |
| 4 | Pickup | Prendre le passager |
| 5 | Dropoff | Déposer le passager |

### 2.4 Système de récompenses par défaut

| Événement | Récompense |
|---|---|
| Chaque step | -1 |
| Dropoff réussi | +20 |
| Pickup ou Dropoff illégal | -10 |

La pénalité de -1 par step incite l'agent à trouver le chemin le plus court. La pénalité de -10 pour les actions illégales décourage les tentatives de pickup/dropoff aléatoires. Le bonus de +20 récompense l'objectif final.

### 2.5 Métriques d'évaluation

Toutes les évaluations sont réalisées sur **100 épisodes de test** avec la même seed (42) pour garantir la comparabilité entre les algorithmes. Les métriques collectées sont :

- **Mean Steps** : nombre moyen d'actions par épisode
- **Mean Reward** : reward cumulé moyen par épisode
- **Success Rate** : pourcentage d'épisodes terminés avec succès (dropoff réussi)
- **Time/episode** : durée moyenne d'un épisode en millisecondes

---

## 3. Algorithme Brute Force

### 3.1 Description

L'agent Brute Force sélectionne une action **uniformément au hasard** à chaque step, sans aucune mémoire ni apprentissage entre les épisodes. Il sert de borne inférieure absolue : tout algorithme RL doit faire mieux.

```
select_action(state) → random.choice([0, 1, 2, 3, 4, 5])
```

### 3.2 Comportement attendu

Sans stratégie, l'agent effectue souvent des actions illégales (pickup/dropoff au mauvais endroit) qui infligent une pénalité de -10 chacune. La probabilité de réussir l'épisode avant le timeout de 200 steps est très faible.

### 3.3 Résultats

| Métrique | Brute Force |
|---|---|
| Mean Steps | 197.2 ± 15.1 |
| Steps min / max | 92 / 200 |
| Mean Reward | -767.43 ± 87.06 |
| Success Rate | 4.0% |
| Time/episode | 1.626 ms |
| Training Episodes | 0 |

> Les épisodes sont quasiment tous tronqués à 200 steps par Gymnasium (4% de réussite par chance pure). Le reward très négatif s'explique par l'accumulation des pénalités de step (-1 × ~197) et des nombreuses actions illégales (-10 chacune) tentées au hasard. Le minimum observé de 92 steps correspond aux rares épisodes où l'agent a eu la chance de réussir par hasard.

### 3.4 Rôle dans le benchmark

Ce résultat établit le plancher de performance. Le passage de Brute Force à Q-Learning représente un gain de **184 steps en moyenne** (197.2 → 13.0) et une transition de 4% à 100% de succès — ce qui quantifie précisément l'apport de l'apprentissage par renforcement sur ce problème.

---

## 4. Q-Learning Tabulaire

### 4.1 Principe

Le Q-Learning est un algorithme **off-policy, model-free** qui maintient une table Q de taille (500 × 6). Pour chaque paire état-action, Q(s, a) représente la valeur espérée du reward cumulé futur si l'on exécute l'action a dans l'état s, puis qu'on suit la politique optimale.

La mise à jour s'effectue à chaque step via la **règle de Bellman** :

```
Q(s, a) ← Q(s, a) + α × [ r + γ × max Q(s', a') - Q(s, a) ]
                                  a'
```

Où :
- **α** (alpha) : taux d'apprentissage — contrôle la vitesse de mise à jour de la Q-table
- **γ** (gamma) : facteur de discount — pondère l'importance des récompenses futures vs immédiates
- **r** : récompense reçue à ce step
- **s'** : état suivant

Le terme `max Q(s', a')` est dit "bootstrapped" : on utilise la valeur courante de la Q-table pour estimer la valeur future, sans modèle de l'environnement. C'est ce qui rend Q-Learning **off-policy** — la cible est calculée avec la politique greedy, indépendamment de la politique d'exploration utilisée.

### 4.2 Exploration : politique epsilon-greedy

Pendant l'entraînement, l'agent suit une politique **epsilon-greedy** :

```
avec probabilité ε  → action aléatoire (exploration)
avec probabilité 1-ε → argmax Q(s, .) (exploitation)
```

ε démarre à 1.0 (exploration totale) et décroît exponentiellement à chaque épisode :

```
ε ← max(ε_min, ε × ε_decay)
```

Cette décroissance assure que l'agent explore suffisamment au début puis converge progressivement vers sa meilleure politique connue. Pendant le test, ε = 0 (greedy pur).

### 4.3 Version non-optimisée (baseline RL)

**Paramètres par défaut :**

| Paramètre | Valeur |
|---|---|
| alpha (α) | 0.1 |
| gamma (γ) | 0.99 |
| epsilon initial | 1.0 |
| epsilon decay | 0.995 |
| epsilon min | 0.01 |
| Épisodes d'entraînement | 1 000 |

**Résultats :**

| Métrique | Q-Learning (défaut) |
|---|---|
| Mean Steps | **13.0 ± 0.0** |
| Mean Reward | **8.00 ± 0.00** |
| Success Rate | **100.0%** |
| Time/episode | 0.104 ms |
| Training Episodes | 1 000 |

**Courbe de convergence (training) :**

| Épisode | Mean Steps | Mean Reward | Success Rate |
|---|---|---|---|
| 200 | 181.6 | -517.10 | 24.0% |
| 400 | 123.4 | -189.84 | 76.0% |
| 600 | 76.5 | -76.45 | 95.0% |
| 800 | 41.8 | -25.81 | 100.0% |
| 1 000 | 26.4 | -6.46 | 100.0% |

> Résultat remarquable : même avec des paramètres par défaut et seulement 1 000 épisodes, la politique greedy au test atteint déjà la solution optimale (13 steps, 100% succès). L'agent atteint 100% de succès dès l'épisode 800. Le decay rapide (0.995) n'est pas pénalisant ici car Taxi-v3 a un espace d'états limité — 1 000 épisodes suffisent à couvrir l'essentiel des états pertinents.

### 4.4 Version optimisée

**Paramètres optimisés :**

| Paramètre | Valeur | Justification |
|---|---|---|
| alpha (α) | 0.8 | Un α élevé permet des mises à jour rapides sur un environnement déterministe |
| gamma (γ) | 0.99 | Valeur proche de 1 : les récompenses futures (dropoff +20) comptent autant que les immédiates |
| epsilon initial | 1.0 | Exploration totale au départ |
| epsilon decay | 0.999 | Décroissance lente : l'agent explore davantage avant de converger |
| epsilon min | 0.01 | 1% d'exploration résiduelle pour ne pas stagner |
| Épisodes d'entraînement | 10 000 | Suffisant pour couvrir les 500 états de manière répétée |

**Résultats :**

| Métrique | Q-Learning (optimisé) |
|---|---|
| Mean Steps | **13.0 ± 0.0** |
| Mean Reward | **8.00 ± 0.00** |
| Success Rate | **100.0%** |
| Time/episode | 0.106 ms |
| Training Episodes | 10 000 |

**Courbe de convergence (training) :**

| Épisode | Mean Steps | Mean Reward | Success Rate |
|---|---|---|---|
| 1 000 | 74.9 | -225.12 | 80.5% |
| 2 000 | 18.1 | -8.62 | 100.0% |
| 4 000 | 13.6 | +6.33 | 100.0% |
| 6 000 | 13.2 | +7.28 | 100.0% |
| 10 000 | 13.3 | +7.30 | 100.0% |

> L'agent optimisé converge plus lentement en début d'entraînement (decay=0.999 plus lent) mais atteint la même performance finale que la version défaut. Sur Taxi-v3, le bénéfice principal du tuning n'est pas la qualité finale mais la **robustesse** : la politique reste stable sur la durée sans risque de sur-exploitation précoce.

**Pourquoi α=0.8 fonctionne bien sur Taxi-v3 ?**

Taxi-v3 est un environnement **déterministe** : la même action dans le même état produit toujours le même résultat. Dans ce cas, un α élevé est bénéfique car il n'y a pas de bruit stochastique à atténuer. L'agent peut se permettre de faire confiance à la dernière observation et mettre à jour sa Q-table agressivement.

**Pourquoi γ=0.99 ?**

La récompense positive (+20) n'arrive qu'à la fin de l'épisode. Avec γ=0.99, cette récompense se propage efficacement en arrière dans la Q-table à travers les mises à jour successives. Un γ plus faible (ex. 0.9) sous-estimerait la valeur des états intermédiaires menant au succès.

### 4.5 Analyse de convergence

La courbe d'apprentissage du Q-Learning optimisé montre trois phases distinctes :

1. **Phase d'exploration (épisodes 0-2000)** : ε élevé, l'agent explore aléatoirement. Les rewards sont très négatifs (-225 à ep 1000) et le succès atteint 80.5% seulement.
2. **Phase de transition (épisodes 2000-4000)** : ε décroît, la Q-table se consolide. Le taux de succès passe à 100% dès l'épisode 2000, les steps descendent rapidement de 18 à 13.
3. **Phase de convergence (épisodes 4000-10000)** : ε proche de ε_min, les performances se stabilisent à 13.2 steps et +7.3 de reward. La politique est optimale.

> **Observation clé** : la version par défaut (α=0.1, 1000 ep) atteint les mêmes 13 steps au test, mais via une trajectoire d'apprentissage plus abrupte et moins stable. Le tuning assure une convergence plus progressive et robuste.

> Graphique de référence : `results/plots/02_steps_evolution.png`

---

## 5. Deep Q-Network (DQN)

### 5.1 Motivation

Bien que Q-Learning tabulaire soit suffisant pour Taxi-v3 (500 états discrets), l'implémentation d'un DQN permet de démontrer une approche scalable à des environnements avec des espaces d'états beaucoup plus grands (images, états continus), où une Q-table serait impossible à maintenir.

### 5.2 Architecture du réseau

Le réseau approxime la fonction Q : **Q(s, a) ≈ QNetwork(encode(s))[a]**

```
Input  : one-hot(state)     → vecteur de taille 500
         (ex: état 42 → [0, 0, ..., 1, ..., 0])

Couche 1 : Linear(500 → 128) + ReLU
Couche 2 : Linear(128 → 64)  + ReLU
Couche 3 : Linear(64  → 6)   (pas d'activation)

Output : Q-valeurs pour les 6 actions → [Q(s,0), Q(s,1), ..., Q(s,5)]
```

L'action choisie est `argmax` de l'output.

**Fonction de loss :** Mean Squared Error (MSE) entre la Q-valeur prédite et la cible :

```
L = MSE( Q_net(s)[a] ,  r + γ · max Q_target(s')[a'] · (1 - done) )
                                   a'
```

### 5.3 Experience Replay

À chaque step, la transition `(s, a, r, s', done)` est stockée dans un **buffer circulaire** de capacité fixe. À chaque mise à jour, un mini-batch est tiré **aléatoirement** depuis ce buffer.

**Pourquoi l'Experience Replay est essentiel ?**

Sans replay, les transitions consécutives sont hautement corrélées (s₀→s₁→s₂ sont temporellement liées). Entraîner un réseau de neurones sur des données corrélées entraîne une instabilité : le réseau "oublie" des états anciens en sur-apprenant les états récents. Le tirage aléatoire brise cette corrélation et stabilise l'apprentissage.

### 5.4 Target Network

Deux réseaux identiques sont maintenus en parallèle :
- **Q-network** : mis à jour à chaque step par descente de gradient
- **Target network** : copie du Q-network, mise à jour toutes les N steps

**Pourquoi un Target Network ?**

Sans target network, les cibles de la loss changent à chaque step (car le même réseau génère à la fois la prédiction et la cible). Cela crée un problème de **"moving targets"** : le réseau chasse une cible qui se déplace en même temps qu'il apprend, ce qui peut mener à des oscillations ou divergences. Le target network fige les cibles pendant N steps, ce qui stabilise considérablement l'entraînement.

### 5.5 Version non-optimisée

**Paramètres par défaut :**

| Paramètre | Valeur |
|---|---|
| Learning rate | 0.001 |
| gamma (γ) | 0.99 |
| epsilon decay | 0.995 |
| Batch size | 64 |
| Buffer capacity | 10 000 |
| Target update freq | 100 steps |
| Épisodes d'entraînement | 3 000 |

**Résultats :**

| Métrique | DQN (défaut) |
|---|---|
| Mean Steps | **13.0 ± 0.0** |
| Mean Reward | **8.00 ± 0.00** |
| Success Rate | **100.0%** |
| Time/episode | 0.845 ms |
| Training Episodes | 3 000 |

**Courbe de convergence (training) :**

| Épisode | Mean Steps | Mean Reward | Success Rate |
|---|---|---|---|
| 300 | 133.3 | -361.26 | 42.0% |
| 600 | 35.5 | -29.87 | 93.7% |
| 900 | 15.3 | +4.43 | 99.3% |
| 1 200 | 14.2 | +6.50 | 100.0% |
| 1 500+ | ~13.4 | ~+7.2 | 100.0% |

### 5.6 Version optimisée

**Paramètres optimisés :**

| Paramètre | Valeur | Justification |
|---|---|---|
| Learning rate | 0.0005 | Plus faible pour éviter l'overshoot sur les Q-valeurs |
| gamma (γ) | 0.99 | Idem Q-Learning |
| epsilon decay | 0.998 | Décroissance plus lente, exploration plus longue |
| Batch size | 64 | Bon compromis variance/vitesse |
| Buffer capacity | 20 000 | Plus grand buffer → diversité accrue des échantillons |
| Target update freq | 200 steps | Cibles plus stables |
| Épisodes d'entraînement | 10 000 | Convergence assurée |

**Résultats :**

| Métrique | DQN (optimisé) |
|---|---|
| Mean Steps | **13.0 ± 0.0** |
| Mean Reward | **8.00 ± 0.00** |
| Success Rate | **100.0%** |
| Time/episode | 0.849 ms |
| Training Episodes | 10 000 |

**Courbe de convergence (training) :**

| Épisode | Mean Steps | Mean Reward | Success Rate |
|---|---|---|---|
| 1 000 | 53.8 | -137.97 | 86.1% |
| 2 000 | 14.1 | +4.48 | 100.0% |
| 3 000 | 13.4 | +7.16 | 100.0% |
| 5 000 | 13.2 | +7.43 | 100.0% |
| 10 000 | 13.2 | +7.37 | 100.0% |

### 5.7 DQN vs Q-Learning sur Taxi-v3

Les deux algorithmes convergent vers la **même politique optimale** (13 steps, 100% succès, reward +8.00), confirmant la théorie sur ce type d'environnement.

| Critère | Q-Learning | DQN |
|---|---|---|
| Performance finale (test) | 13.0 steps, 100% | 13.0 steps, 100% |
| Épisodes pour 100% succès | ~800 (défaut) / ~2000 (tuned) | ~1500 (défaut) / ~2000 (tuned) |
| Time/episode (test) | **0.10 ms** | 0.85 ms (×8.5 plus lent) |
| Interprétabilité | Q-table lisible directement | Réseau opaque |
| Scalabilité | Limitée (500 états) | Illimitée |

Le DQN est **8.5× plus lent** en inférence que Q-Learning sur ce problème, car chaque sélection d'action nécessite un forward pass dans le réseau. Sur Taxi-v3 avec 500 états discrets, le Q-Learning a l'avantage. Le DQN s'imposerait sur une extension avec des milliers d'états ou un espace continu.

> **Conclusion** : Sur Taxi-v3, Q-Learning tabulaire est l'algorithme le plus efficace en termes de vitesse d'entraînement et d'inférence. Le DQN atteint les mêmes performances mais avec un coût computationnel supérieur, justifié uniquement pour des problèmes à plus grande échelle.

---

## 6. Reward Shaping

### 6.1 Motivation

Les récompenses par défaut de Taxi-v3 sont dites **sparses** : le seul signal positif (+20) n'arrive qu'à la toute fin de l'épisode après des dizaines d'actions. En début d'entraînement, l'agent explore quasi-aléatoirement et termine rarement un épisode avec succès, ce qui ralentit la convergence.

Le **reward shaping** consiste à ajouter des récompenses intermédiaires pour guider l'apprentissage sans modifier l'objectif final.

### 6.2 Mode `distance` — Reward basé sur la distance Manhattan

Un bonus proportionnel à la **réduction de distance** est ajouté à chaque step :

```
bonus = (distance_précédente - distance_actuelle) × scale

Phase pickup  : distance = Manhattan(taxi, passager)
Phase dropoff : distance = Manhattan(taxi, destination)
```

- Si le taxi se rapproche de sa cible → bonus positif
- S'il s'en éloigne → bonus négatif (pénalité additionnelle)
- Les récompenses clés (+20 dropoff, -10 illégal) restent inchangées

### 6.3 Résultats comparatifs

Entraînement sur 5 000 épisodes (Q-Learning, α=0.8, γ=0.99, decay=0.999) :

| Mode | Ep 1000 (steps / reward / success) | Ep 5000 (steps / reward / success) |
|---|---|---|
| default  | 73.8 / -220.47 / 82.1% | 13.3 / +7.20 / 100% |
| distance | 69.9 / -202.51 / 83.9% | 13.1 / +11.26 / 100% |

> Graphique de référence : `results/plots/07_reward_shaping.png`

### 6.4 Analyse

Deux observations importantes :

1. **Convergence légèrement plus rapide** avec le mode `distance` (83.9% vs 82.1% de succès à l'épisode 1000). Le signal de reward continu à chaque step guide l'exploration plus efficacement que les récompenses sparses, surtout en début d'entraînement.

2. **Reward cumulé plus élevé** avec `distance` (+11.26 vs +7.20 à convergence). Cela s'explique mécaniquement : les bonus de distance s'accumulent à chaque step, gonflant artificiellement le reward total. En termes de **nombre de steps** (la vraie métrique de performance), les deux modes convergent vers 13 steps identiques.

> **Conclusion** : Le reward shaping `distance` accélère marginalement la convergence sur Taxi-v3, mais n'améliore pas la politique finale. Son intérêt serait plus marqué sur des environnements plus complexes où les récompenses sparses ralentissent davantage l'apprentissage.

---

## 7. Stratégie d'optimisation des hyperparamètres

### 7.1 Méthode : grille de recherche manuelle

Une recherche en grille (*grid search*) a été effectuée sur les hyperparamètres du Q-Learning, en faisant varier deux paramètres clés simultanément : **alpha** et **gamma**.

Pour chaque combinaison :
- 5 000 épisodes d'entraînement
- 50 épisodes de test (seed=42)
- Métrique cible : **mean_steps**

### 7.2 Grille explorée

Mean steps sur 50 épisodes de test (seed=42) après 5 000 épisodes d'entraînement :

| alpha \ gamma | 0.90 | 0.95 | 0.99 |
|---|---|---|---|
| 0.1 | 13.0 | 13.0 | 13.0 |
| 0.3 | 13.0 | 13.0 | 13.0 |
| 0.5 | 13.0 | 13.0 | **15.0** |
| 0.8 | 13.0 | 13.0 | 13.0 |

> Graphique de référence : `results/plots/06_heatmap_alpha_gamma.png`

### 7.3 Observations

- **Robustesse générale** : 11 combinaisons sur 12 atteignent 13.0 steps — l'algorithme Q-Learning est très robuste aux choix d'hyperparamètres sur Taxi-v3 avec 5 000 épisodes d'entraînement.
- **Anomalie (α=0.5, γ=0.99)** : cette combinaison donne 15.0 steps au lieu de 13.0. Ce résultat peut s'expliquer par un hasard de convergence lié à la seed : avec α=0.5 et un gamma élevé, certaines Q-valeurs ont pu être légèrement sous-estimées pour cet état de test particulier.
- **Impact d'alpha** : toutes les valeurs testées convergent aussi bien. Sur un environnement déterministe comme Taxi-v3, un alpha faible (0.1) apprend plus lentement mais finit par couvrir l'espace d'états.
- **Impact de gamma** : γ=0.90 et γ=0.99 donnent les mêmes résultats finals à 5 000 épisodes — le problème est suffisamment simple pour que la propagation des récompenses ne soit pas bloquante.

### 7.4 Choix finaux retenus

| Hyperparamètre | Q-Learning | DQN |
|---|---|---|
| alpha / lr | 0.8 | 0.0005 |
| gamma | 0.99 | 0.99 |
| epsilon decay | 0.999 | 0.998 |
| epsilon min | 0.01 | 0.01 |
| batch size | — | 64 |
| buffer capacity | — | 20 000 |
| target update freq | — | 200 steps |

---

## 8. Tableau comparatif final

> Évaluation sur 100 épisodes de test, seed=42.
> Graphique de référence : `results/plots/04_summary_table.png` et `results/plots/05_steps_distribution.png`

| Algorithme | Train Ep. | Mean Steps | Mean Reward | Success Rate | Time/ep |
|---|---|---|---|---|---|
| Brute Force | 0 | 197.2 ± 15.1 | -767.43 | 4.0% | 1.626 ms |
| Q-Learning (défaut) | 1 000 | **13.0 ± 0.0** | **+8.00** | **100.0%** | 0.104 ms |
| Q-Learning (optimisé) | 10 000 | **13.0 ± 0.0** | **+8.00** | **100.0%** | 0.106 ms |
| DQN (défaut) | 3 000 | **13.0 ± 0.0** | **+8.00** | **100.0%** | 0.845 ms |
| DQN (optimisé) | 10 000 | **13.0 ± 0.0** | **+8.00** | **100.0%** | 0.849 ms |

---

## 9. Conclusion

### 9.1 Synthèse des résultats

Ce projet a permis de comparer plusieurs approches pour résoudre l'environnement Taxi-v3 :

1. **Brute Force** : 197.2 steps en moyenne, 4% de succès. Comme attendu, l'agent aléatoire ne résout presque jamais l'environnement dans la limite des 200 steps. Il constitue le plancher indispensable qui quantifie l'apport du RL.

2. **Q-Learning (défaut)** : résultat surprenant — avec seulement 1 000 épisodes et des paramètres par défaut, l'agent atteint déjà **13.0 steps et 100% de succès** au test. Taxi-v3 est suffisamment simple pour que même des hyperparamètres non-tuned suffisent à trouver la politique optimale.

3. **Q-Learning (optimisé)** : même performance finale (13.0 steps, 100%), mais convergence plus progressive et stable pendant l'entraînement. Le gain du tuning est visible sur la trajectoire d'apprentissage, pas sur le résultat final.

4. **DQN** : atteint les mêmes 13.0 steps et 100% de succès, mais est **8.5× plus lent** en inférence (0.845 ms vs 0.104 ms) et nécessite plus d'épisodes pour converger initialement. Sur Taxi-v3, le DQN n'apporte pas d'avantage pratique sur le Q-Learning tabulaire.

### 9.2 Recommandation

Pour Taxi-v3 spécifiquement, le **Q-Learning tabulaire optimisé** est l'algorithme recommandé :
- Plus rapide à entraîner
- Plus simple à implémenter et à interpréter
- Performances identiques ou légèrement meilleures que le DQN
- La Q-table peut être inspectée directement pour comprendre la politique apprise

Le **DQN** serait l'approche de choix pour une extension du problème (espaces d'états plus grands, environnement continu, ou l'extension 2 passagers).

### 9.3 Limites et perspectives

- **Extension 2 passagers** : l'espace d'états explose combinatoirement. Le Q-Learning tabulaire devient impraticable et le DQN s'impose.
- **Algorithmes alternatifs non testés** : SARSA (on-policy), Monte Carlo ES, PPO (Proximal Policy Optimization) pourraient offrir des perspectives intéressantes.
- **Reward shaping avancé** : des stratégies de shaping plus sophistiquées (potential-based shaping) garantissent de ne pas modifier la politique optimale tout en accélérant la convergence.

---


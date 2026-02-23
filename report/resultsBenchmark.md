(venv) PS C:\Users\teoph\TaxiDriver> python .\src\benchmark.py

=======================================================
  TAXI DRIVER — BENCHMARK COMPLET
  Test : 100 épisodes | seed=42
=======================================================

=======================================================
  RUN 1 — Brute Force (baseline naïf)
=======================================================

[BruteForce] Lancement de 100 épisodes aléatoires...
BruteForce:   0%|                                                                                                                      | 0/100 [00:00<?, ?ep/s][BruteForce] Ep  50/100 | Steps (mean):  195.0 | Reward (mean): -764.18 | Success:   6.0%
BruteForce:  60%|████████████████████████████████████████████████████████████████▊                                           | 60/100 [00:00<00:00, 588.20ep/s][BruteForce] Ep 100/100 | Steps (mean):  199.4 | Reward (mean): -770.68 | Success:   2.0%

===============================
=== Brute Force — Résultats ===
===============================
  Épisodes      : 100
  Steps (mean)  : 197.2  ± 15.1
  Steps (min)   : 92
  Steps (max)   : 200
  Reward (mean) : -767.43  ± 87.06
  Reward (min)  : -929.00
  Reward (max)  : -332.00
  Success rate  : 4.0%
  Time/episode  : 1.626 ms
  Total time    : 0.16 s
===============================

=======================================================
  RUN 2 — Q-Learning (paramètres par défaut)
=======================================================
QLearningAgent(α=0.1, γ=0.99, ε=1.0000, decay=0.995)
Q-Learning Training:  16%|███████████████▍                                                                                 | 159/1000 [00:00<00:01, 515.49ep/s][Training] Ep  200/1000 | Steps (mean):  181.6 | Reward (mean): -517.10 | Success:  24.0%
Q-Learning Training:  39%|█████████████████████████████████████▉                                                           | 391/1000 [00:00<00:00, 706.68ep/s][Training] Ep  400/1000 | Steps (mean):  123.4 | Reward (mean): -189.84 | Success:  76.0%
Q-Learning Training:  52%|██████████████████████████████████████████████████▋                                              | 523/1000 [00:00<00:00, 863.21ep/s][Training] Ep  600/1000 | Steps (mean):   76.5 | Reward (mean):  -76.45 | Success:  95.0%
Q-Learning Training:  70%|███████████████████████████████████████████████████████████████████                             | 698/1000 [00:00<00:00, 1127.94ep/s][Training] Ep  800/1000 | Steps (mean):   41.8 | Reward (mean):  -25.81 | Success: 100.0%
[Training] Ep 1000/1000 | Steps (mean):   26.4 | Reward (mean):   -6.46 | Success: 100.0%
Q-Learning Test:   0%|                                                                                                                 | 0/100 [00:00<?, ?ep/s][Test] Ep  50/100 | Steps (mean):   13.0 | Reward (mean):    8.00 | Success: 100.0%
[Test] Ep 100/100 | Steps (mean):   13.0 | Reward (mean):    8.00 | Success: 100.0%

===================================
=== Q-Learning — Résultats Test ===
===================================
  Épisodes      : 100
  Steps (mean)  : 13.0  ± 0.0
  Steps (min)   : 13
  Steps (max)   : 13
  Reward (mean) : 8.00  ± 0.00
  Reward (min)  : 8.00
  Reward (max)  : 8.00
  Success rate  : 100.0%
  Time/episode  : 0.104 ms
  Total time    : 0.01 s
===================================
[QLearning] Modèle sauvegardé → C:\Users\teoph\TaxiDriver\src\..\results\models\ql_default.pkl

=======================================================
  RUN 3 — Q-Learning (paramètres optimisés)
=======================================================
QLearningAgent(α=0.8, γ=0.99, ε=1.0000, decay=0.999)
Q-Learning Training:   8%|███████▌                                                                                       | 801/10000 [00:00<00:05, 1648.57ep/s][Training] Ep  1000/10000 | Steps (mean):   74.9 | Reward (mean): -225.12 | Success:  80.5%
Q-Learning Training:  17%|███████████████▊                                                                              | 1688/10000 [00:00<00:02, 3143.73ep/s][Training] Ep  2000/10000 | Steps (mean):   18.1 | Reward (mean):   -8.62 | Success: 100.0%
Q-Learning Training:  29%|███████████████████████████                                                                   | 2885/10000 [00:01<00:01, 4534.17ep/s][Training] Ep  3000/10000 | Steps (mean):   14.5 | Reward (mean):    2.91 | Success: 100.0%
Q-Learning Training:  34%|████████████████████████████████                                                              | 3409/10000 [00:01<00:01, 4732.64ep/s][Training] Ep  4000/10000 | Steps (mean):   13.6 | Reward (mean):    6.33 | Success: 100.0%
Q-Learning Training:  47%|████████████████████████████████████████████▎                                                 | 4714/10000 [00:01<00:00, 5568.56ep/s][Training] Ep  5000/10000 | Steps (mean):   13.4 | Reward (mean):    7.14 | Success: 100.0%
Q-Learning Training:  54%|███████████████████████████████████████████████████                                           | 5437/10000 [00:01<00:00, 5947.12ep/s][Training] Ep  6000/10000 | Steps (mean):   13.2 | Reward (mean):    7.28 | Success: 100.0%
Q-Learning Training:  70%|█████████████████████████████████████████████████████████████████▎                            | 6951/10000 [00:01<00:00, 6532.59ep/s][Training] Ep  7000/10000 | Steps (mean):   13.2 | Reward (mean):    7.41 | Success: 100.0%
Q-Learning Training:  77%|████████████████████████████████████████████████████████████████████████                      | 7666/10000 [00:01<00:00, 6674.23ep/s][Training] Ep  8000/10000 | Steps (mean):   13.2 | Reward (mean):    7.41 | Success: 100.0%
Q-Learning Training:  84%|██████████████████████████████████████████████████████████████████████████████▍               | 8350/10000 [00:01<00:00, 6716.58ep/s][Training] Ep  9000/10000 | Steps (mean):   13.3 | Reward (mean):    7.36 | Success: 100.0%
Q-Learning Training:  98%|███████████████████████████████████████████████████████████████████████████████████████████▋  | 9757/10000 [00:02<00:00, 6840.54ep/s][Training] Ep 10000/10000 | Steps (mean):   13.3 | Reward (mean):    7.30 | Success: 100.0%
Q-Learning Test:   0%|                                                                                                                 | 0/100 [00:00<?, ?ep/s][Test] Ep  50/100 | Steps (mean):   13.0 | Reward (mean):    8.00 | Success: 100.0%
[Test] Ep 100/100 | Steps (mean):   13.0 | Reward (mean):    8.00 | Success: 100.0%

===================================
=== Q-Learning — Résultats Test ===
===================================
  Épisodes      : 100
  Steps (mean)  : 13.0  ± 0.0
  Steps (min)   : 13
  Steps (max)   : 13
  Reward (mean) : 8.00  ± 0.00
  Reward (min)  : 8.00
  Reward (max)  : 8.00
  Success rate  : 100.0%
  Time/episode  : 0.106 ms
  Total time    : 0.01 s
===================================
[QLearning] Modèle sauvegardé → C:\Users\teoph\TaxiDriver\src\..\results\models\ql_tuned.pkl

=======================================================
  RUN 4 — DQN (paramètres par défaut)
=======================================================
DQNAgent(lr=0.001, γ=0.99, ε=1.0000, batch=64, target_freq=100, device=cpu)
DQN Training:  10%|██████████▍                                                                                              | 297/3000 [01:34<05:49,  7.74ep/s][Training] Ep  300/3000 | Steps (mean):  133.3 | Reward (mean): -361.26 | Success:  42.0%
DQN Training:  20%|████████████████████▉                                                                                    | 597/3000 [02:00<01:23, 28.67ep/s][Training] Ep  600/3000 | Steps (mean):   35.5 | Reward (mean):  -29.87 | Success:  93.7%
DQN Training:  30%|███████████████████████████████▍                                                                         | 899/3000 [02:11<01:06, 31.53ep/s][Training] Ep  900/3000 | Steps (mean):   15.3 | Reward (mean):    4.43 | Success:  99.3%
DQN Training:  40%|█████████████████████████████████████████▌                                                              | 1199/3000 [02:21<00:58, 30.78ep/s][Training] Ep 1200/3000 | Steps (mean):   14.2 | Reward (mean):    6.50 | Success: 100.0%
DQN Training:  50%|███████████████████████████████████████████████████▉                                                    | 1498/3000 [02:32<00:52, 28.77ep/s][Training] Ep 1500/3000 | Steps (mean):   13.3 | Reward (mean):    7.55 | Success: 100.0%
DQN Training:  60%|██████████████████████████████████████████████████████████████▎                                         | 1797/3000 [02:42<00:40, 29.77ep/s][Training] Ep 1800/3000 | Steps (mean):   13.6 | Reward (mean):    7.15 | Success: 100.0%
DQN Training:  70%|████████████████████████████████████████████████████████████████████████▋                               | 2097/3000 [02:53<00:29, 30.90ep/s][Training] Ep 2100/3000 | Steps (mean):   13.8 | Reward (mean):    6.85 | Success: 100.0%
DQN Training:  80%|███████████████████████████████████████████████████████████████████████████████████                     | 2397/3000 [03:04<00:23, 25.58ep/s][Training] Ep 2400/3000 | Steps (mean):   13.7 | Reward (mean):    7.00 | Success: 100.0%
DQN Training:  90%|█████████████████████████████████████████████████████████████████████████████████████████████▌          | 2699/3000 [03:15<00:10, 27.62ep/s][Training] Ep 2700/3000 | Steps (mean):   13.7 | Reward (mean):    6.86 | Success: 100.0%
DQN Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████▉| 2997/3000 [03:25<00:00, 32.86ep/s][Training] Ep 3000/3000 | Steps (mean):   13.6 | Reward (mean):    7.15 | Success: 100.0%
DQN Test:   0%|                                                                                                                        | 0/100 [00:00<?, ?ep/s][Test] Ep  50/100 | Steps (mean):   13.0 | Reward (mean):    8.00 | Success: 100.0%
[Test] Ep 100/100 | Steps (mean):   13.0 | Reward (mean):    8.00 | Success: 100.0%

============================
=== DQN — Résultats Test ===
============================
  Épisodes      : 100
  Steps (mean)  : 13.0  ± 0.0
  Steps (min)   : 13
  Steps (max)   : 13
  Reward (mean) : 8.00  ± 0.00
  Reward (min)  : 8.00
  Reward (max)  : 8.00
  Success rate  : 100.0%
  Time/episode  : 0.845 ms
  Total time    : 0.08 s
============================
[DQN] Modèle sauvegardé → C:\Users\teoph\TaxiDriver\src\..\results\models\dqn_default.pt

=======================================================
  RUN 5 — DQN (paramètres optimisés)
=======================================================
DQNAgent(lr=0.0005, γ=0.99, ε=1.0000, batch=64, target_freq=200, device=cpu)
DQN Training:  10%|██████████▍                                                                                             | 999/10000 [02:07<05:47, 25.87ep/s][Training] Ep  1000/10000 | Steps (mean):   53.8 | Reward (mean): -137.97 | Success:  86.1%
DQN Training:  20%|████████████████████▌                                                                                  | 1998/10000 [02:42<04:27, 29.96ep/s][Training] Ep  2000/10000 | Steps (mean):   14.1 | Reward (mean):    4.48 | Success: 100.0%
DQN Training:  30%|██████████████████████████████▊                                                                        | 2996/10000 [03:16<03:47, 30.75ep/s][Training] Ep  3000/10000 | Steps (mean):   13.4 | Reward (mean):    7.16 | Success: 100.0%
DQN Training:  40%|█████████████████████████████████████████▏                                                             | 3999/10000 [04:04<10:12,  9.80ep/s][Training] Ep  4000/10000 | Steps (mean):   13.5 | Reward (mean):    7.08 | Success: 100.0%
DQN Training:  50%|███████████████████████████████████████████████████▍                                                   | 4999/10000 [04:47<03:11, 26.14ep/s][Training] Ep  5000/10000 | Steps (mean):   13.2 | Reward (mean):    7.43 | Success: 100.0%
DQN Training:  60%|█████████████████████████████████████████████████████████████▊                                         | 5998/10000 [05:24<02:23, 27.94ep/s][Training] Ep  6000/10000 | Steps (mean):   13.3 | Reward (mean):    7.25 | Success: 100.0%
DQN Training:  70%|████████████████████████████████████████████████████████████████████████                               | 6999/10000 [05:59<01:53, 26.41ep/s][Training] Ep  7000/10000 | Steps (mean):   13.2 | Reward (mean):    7.49 | Success: 100.0%
DQN Training:  80%|██████████████████████████████████████████████████████████████████████████████████▍                    | 7999/10000 [06:36<01:11, 27.91ep/s][Training] Ep  8000/10000 | Steps (mean):   13.1 | Reward (mean):    7.54 | Success: 100.0%
DQN Training:  90%|████████████████████████████████████████████████████████████████████████████████████████████▋          | 8998/10000 [07:14<00:36, 27.43ep/s][Training] Ep  9000/10000 | Steps (mean):   13.2 | Reward (mean):    7.54 | Success: 100.0%
DQN Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████▉| 9999/10000 [07:50<00:00, 28.44ep/s][Training] Ep 10000/10000 | Steps (mean):   13.2 | Reward (mean):    7.37 | Success: 100.0%
DQN Test:   0%|                                                                                                                        | 0/100 [00:00<?, ?ep/s][Test] Ep  50/100 | Steps (mean):   13.0 | Reward (mean):    8.00 | Success: 100.0%
[Test] Ep 100/100 | Steps (mean):   13.0 | Reward (mean):    8.00 | Success: 100.0%

============================
=== DQN — Résultats Test ===
============================
  Épisodes      : 100
  Steps (mean)  : 13.0  ± 0.0
  Steps (min)   : 13
  Steps (max)   : 13
  Reward (mean) : 8.00  ± 0.00
  Reward (min)  : 8.00
  Reward (max)  : 8.00
  Success rate  : 100.0%
  Time/episode  : 0.849 ms
  Total time    : 0.08 s
============================
[DQN] Modèle sauvegardé → C:\Users\teoph\TaxiDriver\src\..\results\models\dqn_tuned.pt

=======================================================
  RUN 6 — Comparaison Reward Shaping
=======================================================

  → reward_mode='default'
Q-Learning Training:  14%|█████████████▌                                                                                  | 706/5000 [00:00<00:02, 1472.19ep/s][Training] Ep 1000/5000 | Steps (mean):   73.8 | Reward (mean): -220.47 | Success:  82.1%
Q-Learning Training:  34%|███████████████████████████████▉                                                               | 1681/5000 [00:00<00:01, 3204.14ep/s][Training] Ep 2000/5000 | Steps (mean):   17.8 | Reward (mean):   -8.15 | Success: 100.0%
Q-Learning Training:  57%|██████████████████████████████████████████████████████▍                                        | 2865/5000 [00:01<00:00, 4541.80ep/s][Training] Ep 3000/5000 | Steps (mean):   14.5 | Reward (mean):    2.97 | Success: 100.0%
Q-Learning Training:  70%|██████████████████████████████████████████████████████████████████▋                            | 3511/5000 [00:01<00:00, 5030.93ep/s][Training] Ep 4000/5000 | Steps (mean):   13.5 | Reward (mean):    6.68 | Success: 100.0%
Q-Learning Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████▌| 4978/5000 [00:01<00:00, 5994.29ep/s][Training] Ep 5000/5000 | Steps (mean):   13.3 | Reward (mean):    7.20 | Success: 100.0%

  → reward_mode='distance'
Q-Learning Training:  16%|███████████████▋                                                                                | 817/5000 [00:00<00:02, 1697.17ep/s][Training] Ep 1000/5000 | Steps (mean):   69.9 | Reward (mean): -202.51 | Success:  83.9%
Q-Learning Training:  34%|████████████████████████████████▍                                                              | 1705/5000 [00:00<00:01, 3164.68ep/s][Training] Ep 2000/5000 | Steps (mean):   17.9 | Reward (mean):   -4.01 | Success: 100.0%
Q-Learning Training:  58%|███████████████████████████████████████████████████████                                        | 2900/5000 [00:01<00:00, 4536.17ep/s][Training] Ep 3000/5000 | Steps (mean):   14.5 | Reward (mean):    6.91 | Success: 100.0%
Q-Learning Training:  70%|██████████████████████████████████████████████████████████████████▋                            | 3513/5000 [00:01<00:00, 5000.91ep/s][Training] Ep 4000/5000 | Steps (mean):   13.6 | Reward (mean):   10.13 | Success: 100.0%
Q-Learning Training:  96%|███████████████████████████████████████████████████████████████████████████████████████████▌   | 4821/5000 [00:01<00:00, 5721.13ep/s][Training] Ep 5000/5000 | Steps (mean):   13.1 | Reward (mean):   11.26 | Success: 100.0%
[Viz] Graphique sauvegardé → C:\Users\teoph\TaxiDriver\src\utils\..\..\results\plots\07_reward_shaping.png

=======================================================
  RUN 7 — Recherche hyperparamètres (alpha × gamma)
=======================================================
                                                                                                                                                                
===================================
=== Q-Learning — Résultats Test ===
===================================
  Épisodes      : 50
  Steps (mean)  : 13.0  ± 0.0
  Steps (min)   : 13
  Steps (max)   : 13
  Reward (mean) : 8.00  ± 0.00
  Reward (min)  : 8.00
  Reward (max)  : 8.00
  Success rate  : 100.0%
  Time/episode  : 0.109 ms
  Total time    : 0.01 s
===================================
  mean_steps=13.0
                                                                                                                                                                
===================================
=== Q-Learning — Résultats Test ===
===================================
  Épisodes      : 50
  Steps (mean)  : 13.0  ± 0.0
  Steps (min)   : 13
  Steps (max)   : 13
  Reward (mean) : 8.00  ± 0.00
  Reward (min)  : 8.00
  Reward (max)  : 8.00
  Success rate  : 100.0%
  Time/episode  : 0.102 ms
  Total time    : 0.01 s
===================================
  mean_steps=13.0

===================================
=== Q-Learning — Résultats Test ===
===================================
  Épisodes      : 50
  Steps (mean)  : 13.0  ± 0.0
  Steps (min)   : 13
  Steps (max)   : 13
  Reward (mean) : 8.00  ± 0.00
  Reward (min)  : 8.00
  Reward (max)  : 8.00
  Success rate  : 100.0%
  Time/episode  : 0.089 ms
  Total time    : 0.00 s
===================================
  mean_steps=13.0
                                                                                                                                                                
===================================
=== Q-Learning — Résultats Test ===
===================================
  Épisodes      : 50
  Steps (mean)  : 13.0  ± 0.0
  Steps (min)   : 13
  Steps (max)   : 13
  Reward (mean) : 8.00  ± 0.00
  Reward (min)  : 8.00
  Reward (max)  : 8.00
  Success rate  : 100.0%
  Time/episode  : 0.091 ms
  Total time    : 0.00 s
===================================
  mean_steps=13.0
                                                                                                                                                                
===================================
=== Q-Learning — Résultats Test ===
===================================
  Épisodes      : 50
  Steps (mean)  : 13.0  ± 0.0
  Steps (min)   : 13
  Steps (max)   : 13
  Reward (mean) : 8.00  ± 0.00
  Reward (min)  : 8.00
  Reward (max)  : 8.00
  Success rate  : 100.0%
  Time/episode  : 0.092 ms
  Total time    : 0.00 s
===================================
  mean_steps=13.0

===================================
=== Q-Learning — Résultats Test ===
===================================
  Épisodes      : 50
  Steps (mean)  : 13.0  ± 0.0
  Steps (min)   : 13
  Steps (max)   : 13
  Reward (mean) : 8.00  ± 0.00
  Reward (min)  : 8.00
  Reward (max)  : 8.00
  Success rate  : 100.0%
  Time/episode  : 0.103 ms
  Total time    : 0.01 s
===================================
  mean_steps=13.0

===================================
=== Q-Learning — Résultats Test ===
===================================
  Épisodes      : 50
  Steps (mean)  : 13.0  ± 0.0
  Steps (min)   : 13
  Steps (max)   : 13
  Reward (mean) : 8.00  ± 0.00
  Reward (min)  : 8.00
  Reward (max)  : 8.00
  Success rate  : 100.0%
  Time/episode  : 0.091 ms
  Total time    : 0.00 s
===================================
  mean_steps=13.0

===================================
=== Q-Learning — Résultats Test ===
===================================
  Épisodes      : 50
  Steps (mean)  : 13.0  ± 0.0
  Steps (min)   : 13
  Steps (max)   : 13
  Reward (mean) : 8.00  ± 0.00
  Reward (min)  : 8.00
  Reward (max)  : 8.00
  Success rate  : 100.0%
  Time/episode  : 0.117 ms
  Total time    : 0.01 s
===================================
  mean_steps=13.0
                                                                                                                                                                
===================================
=== Q-Learning — Résultats Test ===
===================================
  Épisodes      : 50
  Steps (mean)  : 15.0  ± 0.0
  Steps (min)   : 15
  Steps (max)   : 15
  Reward (mean) : 6.00  ± 0.00
  Reward (min)  : 6.00
  Reward (max)  : 6.00
  Success rate  : 100.0%
  Time/episode  : 0.108 ms
  Total time    : 0.01 s
===================================
  mean_steps=15.0

===================================
=== Q-Learning — Résultats Test ===
===================================
  Épisodes      : 50
  Steps (mean)  : 13.0  ± 0.0
  Steps (min)   : 13
  Steps (max)   : 13
  Reward (mean) : 8.00  ± 0.00
  Reward (min)  : 8.00
  Reward (max)  : 8.00
  Success rate  : 100.0%
  Time/episode  : 0.104 ms
  Total time    : 0.01 s
===================================
  mean_steps=13.0
                                                                                                                                                                
===================================
=== Q-Learning — Résultats Test ===
===================================
  Épisodes      : 50
  Steps (mean)  : 13.0  ± 0.0
  Steps (min)   : 13
  Steps (max)   : 13
  Reward (mean) : 8.00  ± 0.00
  Reward (min)  : 8.00
  Reward (max)  : 8.00
  Success rate  : 100.0%
  Time/episode  : 0.104 ms
  Total time    : 0.01 s
===================================
  mean_steps=13.0
                                                                                                                                                                
===================================
=== Q-Learning — Résultats Test ===
===================================
  Épisodes      : 50
  Steps (mean)  : 13.0  ± 0.0
  Steps (min)   : 13
  Steps (max)   : 13
  Reward (mean) : 8.00  ± 0.00
  Reward (min)  : 8.00
  Reward (max)  : 8.00
  Success rate  : 100.0%
  Time/episode  : 0.099 ms
  Total time    : 0.00 s
===================================
  mean_steps=13.0
[Viz] Graphique sauvegardé → C:\Users\teoph\TaxiDriver\src\utils\..\..\results\plots\06_heatmap_alpha_gamma.png
[Viz] Graphique sauvegardé → C:\Users\teoph\TaxiDriver\src\utils\..\..\results\plots\06_heatmap_alpha_gamma.png

=======================================================
  GRAPHIQUES COMPARATIFS FINAUX
=======================================================
[Viz] Graphique sauvegardé → C:\Users\teoph\TaxiDriver\src\utils\..\..\results\plots\01_learning_curves.png
[Viz] Graphique sauvegardé → C:\Users\teoph\TaxiDriver\src\utils\..\..\results\plots\02_steps_evolution.png
[Viz] Graphique sauvegardé → C:\Users\teoph\TaxiDriver\src\utils\..\..\results\plots\03_epsilon_decay.png
C:\Users\teoph\TaxiDriver\src\utils\visualization.py:233: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.boxplot(
[Viz] Graphique sauvegardé → C:\Users\teoph\TaxiDriver\src\utils\..\..\results\plots\05_steps_distribution.png
[Viz] Graphique sauvegardé → C:\Users\teoph\TaxiDriver\src\utils\..\..\results\plots\04_summary_table.png

          Algorithme Mean Steps Std Steps Mean Reward Success Rate Time/ep (ms)
          BruteForce      197.2    ± 15.1     -767.43         4.0%         1.63
Q-Learning (default)       13.0     ± 0.0        8.00       100.0%         0.10
  Q-Learning (tuned)       13.0     ± 0.0        8.00       100.0%         0.11
       DQN (default)       13.0     ± 0.0        8.00       100.0%         0.85
         DQN (tuned)       13.0     ± 0.0        8.00       100.0%         0.85

=======================================================
  Benchmark terminé. Graphiques dans results/plots/
=======================================================
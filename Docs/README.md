# TaxiDriver - Documentation

Ce dossier contient la documentation technique du projet TaxiDriver.

## Sommaire

1. [Vue d'ensemble](./architecture.md)
2. [Flux d'execution](./workflows.md)
3. [Documentation HTML](./index.html)

## Objectif du projet

Le projet compare trois approches sur `Taxi-v3`:

- `BruteForceAgent`: baseline aleatoire
- `QLearningAgent`: apprentissage tabulaire
- `DQNAgent`: reseau de neurones + replay buffer + target network

## Lancer le projet

```bash
python src/main.py
```

Benchmark complet:

```bash
python src/benchmark.py
```

## Sorties generees

- Modeles: `results/models/`
- Graphiques: `results/plots/`
- Rapport: `report/report.md`

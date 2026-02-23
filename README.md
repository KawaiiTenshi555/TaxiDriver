# Taxi Driver — T-AIA-902

Reinforcement Learning project solving the `Taxi-v3` environment from Gymnasium.

An agent (taxi) learns to pick up a passenger at a random location and drop them off at the correct destination in a minimum number of steps.

## Algorithms

- **Brute Force** — random agent (naive baseline)
- **Q-Learning** — tabular, model-free, off-policy (RL baseline)
- **Deep Q-Network (DQN)** — neural network, experience replay, target network (main algorithm)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py
```

Two modes are available at launch:

- **User mode** — manually configure all algorithm hyperparameters
- **Time-limited mode** — runs with optimized parameters within a given time budget

Both training and testing episode counts are entered interactively at startup.

## Output

- Mean steps and mean reward per episode
- Success rate over test episodes
- Random episode replays in the console
- Benchmark graphs saved in `results/plots/`

## Project Structure

```
TaxiDriver/
├── src/
│   ├── agents/          # Brute force, Q-Learning, DQN
│   ├── environment/     # Gymnasium wrapper + reward shaping
│   ├── utils/           # Metrics, visualization, replay buffer
│   └── main.py
├── results/
│   ├── models/          # Saved trained models
│   └── plots/           # Generated graphs
├── report/              # Benchmarking report
├── requirements.txt
└── README.md
```

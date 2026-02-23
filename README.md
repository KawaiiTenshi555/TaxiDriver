# Taxi Driver — T-AIA-902

Reinforcement Learning project solving the `Taxi-v3` environment from Gymnasium.

An agent (taxi) learns to pick up a passenger at a random location and drop them off at the correct destination in a minimum number of steps.

## Algorithms

| Algorithm | Type | Role |
|---|---|---|
| **Brute Force** | Random agent | Naive baseline (~200 steps, 0% success) |
| **Q-Learning** | Tabular, off-policy | RL baseline |
| **DQN** | Neural network + Experience Replay + Target Network | Main algorithm |

## Installation

```bash
python -m venv venv
source venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

## Usage

### Interactive program (two modes)

```bash
python src/main.py
```

**User mode** — manually configure algorithm, hyperparameters, reward shaping, and episode counts.

**Time-limited mode** — pre-optimized parameters, train for a given number of seconds then evaluate.

### Full benchmark (all algorithms)

```bash
python src/benchmark.py
```

Runs all 7 benchmark runs sequentially (Brute Force → Q-Learning default → Q-Learning tuned → DQN default → DQN tuned → Reward shaping comparison → Hyperparameter heatmap) and saves all graphs to `results/plots/`.

### Validation scripts

```bash
python scripts/test_bruteforce.py   # Quick brute force validation
python scripts/test_qlearning.py    # Q-Learning default vs optimized
```

## Output

- Mean steps and mean reward per episode (training + test)
- Success rate over test episodes
- Random episode replays in the console (ASCII)
- All benchmark graphs saved in `results/plots/`
- Trained models saved in `results/models/`

## Benchmark Results

> See `results/plots/` for all graphs and `report/report.md` for full analysis.

| Algorithm | Train Episodes | Mean Steps | Success Rate |
|---|---|---|---|
| Brute Force | 0 | ~200 | ~0% |
| Q-Learning (default) | 1 000 | — | — |
| Q-Learning (tuned) | 10 000 | — | — |
| DQN (default) | 3 000 | — | — |
| DQN (tuned) | 10 000 | — | — |

*Fill after running `python src/benchmark.py`*

## Project Structure

```
TaxiDriver/
├── src/
│   ├── agents/
│   │   ├── brute_force.py     # Random agent
│   │   ├── q_learning.py      # Tabular Q-Learning
│   │   └── dqn.py             # Deep Q-Network
│   ├── environment/
│   │   └── taxi_wrapper.py    # Gymnasium wrapper + reward shaping
│   ├── utils/
│   │   ├── metrics.py         # Episode tracking + statistics
│   │   ├── visualization.py   # All benchmark graphs (7 types)
│   │   └── replay_buffer.py   # Experience replay for DQN
│   ├── main.py                # Entry point (User / Time-Limited modes)
│   └── benchmark.py           # Full benchmark script
├── scripts/
│   ├── test_bruteforce.py     # Brute force validation
│   └── test_qlearning.py      # Q-Learning validation
├── results/
│   ├── models/                # Saved trained models (.pkl, .pt)
│   └── plots/                 # Generated graphs (.png)
├── report/
│   └── report.md              # Full benchmarking report
├── requirements.txt
└── README.md
```

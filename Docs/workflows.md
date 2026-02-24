# Flux D'execution

## 1) Entrainement standard (`main.py`)

```mermaid
sequenceDiagram
    participant User
    participant Main as main.py
    participant Agent
    participant Env as TaxiWrapper
    participant Metrics as MetricsTracker

    User->>Main: Choix mode + hyperparametres
    Main->>Agent: Instanciation
    loop Chaque episode
        Main->>Env: reset()
        Main->>Metrics: begin_episode()
        loop Chaque step
            Agent->>Env: step(action)
            Env-->>Agent: next_state, reward, done
            Agent->>Agent: update() / learn()
        end
        Agent->>Agent: decay epsilon
        Main->>Metrics: end_episode()
    end
    Main->>Agent: test()
    Main->>Main: replay episodes
    Main->>Main: generation des graphes
```

## 2) Boucle Q-Learning

```mermaid
flowchart TD
    S[Etat s] --> A[Choix action a epsilon-greedy]
    A --> E[Env.step(a)]
    E --> R[Recoit r, s']
    R --> U[Mise a jour Bellman]
    U --> D{done ?}
    D -- non --> S
    D -- oui --> EP[Fin episode + decay epsilon]
```

Formule:

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
```

## 3) Boucle DQN

```mermaid
flowchart TD
    S[Etat s] --> A[Action epsilon-greedy]
    A --> E[Env.step]
    E --> P[Push transition dans ReplayBuffer]
    P --> B{Buffer pret ?}
    B -- non --> C[Step suivant]
    B -- oui --> L[Sample mini-batch]
    L --> T[Calcul target avec target_net]
    T --> G[Backprop sur q_net]
    G --> Y{Sync target ?}
    Y -- oui --> Z[Copie q_net -> target_net]
    Y -- non --> C
    Z --> C
```

## 4) Reward shaping dans `TaxiWrapper`

```mermaid
flowchart LR
    M[reward_mode] --> D[default]
    M --> DS[distance]
    M --> C[custom]
    DS --> B[bonus selon reduction distance Manhattan]
    C --> MAP[mapping custom step/dropoff/illegal]
```

## 5) Generation de resultats

1. Les agents produisent des `MetricsTracker`.
2. `visualization.py` transforme les metriques en figures PNG.
3. Les figures sont enregistrees dans `results/plots/`.

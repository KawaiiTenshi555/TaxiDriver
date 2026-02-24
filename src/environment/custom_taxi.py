"""Custom Taxi environment: variable grid size + multiple passengers."""

from __future__ import annotations

import numpy as np


# ── Location computation ────────────────────────────────────────────────────

def _compute_locations(grid_size: int) -> list[tuple[int, int]]:
    """Return pickup/dropoff locations for a given grid size.

    Always includes the 4 corners. For larger grids, midpoints are added
    so there are enough distinct locations for more passengers.
    """
    G = grid_size
    locs: list[tuple[int, int]] = [
        (0,     0),      # corner TL
        (0,     G - 1),  # corner TR
        (G - 1, 0),      # corner BL
        (G - 1, G - 1),  # corner BR
    ]
    if G >= 6:
        locs.append((0,     G // 2))   # top mid
        locs.append((G - 1, G // 2))   # bottom mid
    if G >= 9:
        locs.append((G // 2, 0))       # left mid
        locs.append((G // 2, G - 1))   # right mid
    return locs


def max_passengers_for_grid(grid_size: int) -> int:
    """Return the maximum number of passengers supported by a given grid size."""
    return len(_compute_locations(grid_size)) - 1


# ── Custom environment ──────────────────────────────────────────────────────

class CustomTaxiEnv:
    """
    Taxi environment with configurable grid size and number of passengers.

    Rules:
    - Taxi moves N/S/E/W, picks up one passenger at a time, delivers to destination.
    - Each passenger i starts at location i (fixed pickup spot).
    - Destinations are randomly assigned at episode start.
    - Episode terminates when all passengers are delivered.
    - Truncated after max_steps steps (default: 200 * n_passengers).

    State encoding (integer):
        state = taxi_pos * (3^P * L^P) + stat_enc * L^P + dest_enc
        where L = n_locations, P = n_passengers.

    Feature vector for DQN (compact):
        [taxi_row/(G-1), taxi_col/(G-1),
         *one_hot_status_P0(3), *one_hot_dest_P0(L),
         *one_hot_status_P1(3), *one_hot_dest_P1(L), ...]
        Size: 2 + n_passengers * (3 + n_locations)
    """

    def __init__(
        self,
        grid_size: int = 5,
        n_passengers: int = 1,
        reward_mode: str = "default",
        custom_rewards: dict | None = None,
        max_steps: int | None = None,
    ) -> None:
        if grid_size < 3 or grid_size > 12:
            raise ValueError(f"grid_size doit être entre 3 et 12, reçu {grid_size}")

        self.grid_size = grid_size
        self.n_passengers = n_passengers
        self.n_actions = 6
        self.reward_mode = reward_mode

        # Locations
        all_locs = _compute_locations(grid_size)
        max_p = len(all_locs) - 1
        if n_passengers < 1 or n_passengers > max_p:
            raise ValueError(
                f"Grille {grid_size}×{grid_size} supporte 1 à {max_p} passagers, "
                f"reçu {n_passengers}"
            )
        self.locations: list[tuple[int, int]] = all_locs
        self.n_locations = len(self.locations)

        # State space dimensions
        G, P, L = grid_size, n_passengers, self.n_locations
        self.n_states = G * G * (3 ** P) * (L ** P)
        self.n_state_features = 2 + P * (3 + L)

        # Rewards
        self._rewards: dict[str, float] = {
            "step":            -1.0,
            "dropoff_success": +20.0,
            "illegal_action":  -10.0,
        }
        if custom_rewards:
            self._rewards.update(custom_rewards)

        # Max steps per episode
        self._max_steps = max_steps if max_steps is not None else 200 * n_passengers

        # RNG & internal state
        self._rng = np.random.default_rng()
        self._taxi_row   = 0
        self._taxi_col   = 0
        self._statuses: list[int] = [0] * n_passengers  # 0=waiting, 1=in_taxi, 2=delivered
        self._dests:    list[int] = [0] * n_passengers  # destination location index
        self._step_count = 0

    # ------------------------------------------------------------------
    # Core API  (compatible with TaxiWrapper)
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[int, dict]:
        """Reset episode. Returns (state_int, info)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        G, P, L = self.grid_size, self.n_passengers, self.n_locations

        self._taxi_row = int(self._rng.integers(0, G))
        self._taxi_col = int(self._rng.integers(0, G))
        self._statuses = [0] * P

        self._dests = []
        for i in range(P):
            possible = [j for j in range(L) if j != i]
            self._dests.append(int(self._rng.choice(possible)))

        self._step_count = 0
        return self._encode(), {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        """Execute action. Returns (next_state, reward, terminated, truncated, info)."""
        G = self.grid_size
        reward: float = self._rewards["step"]
        terminated = False

        if action == 0:    # South
            self._taxi_row = min(G - 1, self._taxi_row + 1)
        elif action == 1:  # North
            self._taxi_row = max(0,     self._taxi_row - 1)
        elif action == 2:  # East
            self._taxi_col = min(G - 1, self._taxi_col + 1)
        elif action == 3:  # West
            self._taxi_col = max(0,     self._taxi_col - 1)
        elif action == 4:  # Pickup
            reward = self._do_pickup()
        elif action == 5:  # Dropoff
            reward = self._do_dropoff()

        if all(s == 2 for s in self._statuses):
            terminated = True

        self._step_count += 1
        truncated = (not terminated) and (self._step_count >= self._max_steps)

        return self._encode(), reward, terminated, truncated, {}

    def close(self) -> None:
        pass

    def render(self) -> str:
        """Simple ASCII rendering of the grid state."""
        G = self.grid_size
        loc_names = "RGYBCDEF"  # up to 8 locations
        rows = ["+" + "-" * (2 * G - 1) + "+"]
        for r in range(G):
            cells = []
            for c in range(G):
                ch = " "
                for li, pos in enumerate(self.locations):
                    if pos == (r, c):
                        ch = loc_names[li] if li < len(loc_names) else "?"
                        break
                if (r, c) == (self._taxi_row, self._taxi_col):
                    ch = "P" if any(s == 1 for s in self._statuses) else "T"
                cells.append(ch)
            rows.append("|" + " ".join(cells) + "|")
        rows.append("+" + "-" * (2 * G - 1) + "+")
        # Status line
        status_str = " | ".join(
            f"P{i}:{'wait' if s == 0 else 'taxi' if s == 1 else 'done'}"
            for i, s in enumerate(self._statuses)
        )
        rows.append(f"[{status_str}]")
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Pickup / Dropoff logic
    # ------------------------------------------------------------------

    def _do_pickup(self) -> float:
        taxi_pos = (self._taxi_row, self._taxi_col)
        # Cannot pick up if someone is already in the taxi
        if any(s == 1 for s in self._statuses):
            return self._rewards["illegal_action"]
        # Find a waiting passenger at taxi's position
        for i, status in enumerate(self._statuses):
            if status == 0 and self.locations[i] == taxi_pos:
                self._statuses[i] = 1
                return self._rewards["step"]   # successful pickup costs only 1 step
        return self._rewards["illegal_action"]

    def _do_dropoff(self) -> float:
        taxi_pos = (self._taxi_row, self._taxi_col)
        for i, status in enumerate(self._statuses):
            if status == 1:   # found the passenger in the taxi
                dest_pos = self.locations[self._dests[i]]
                if taxi_pos == dest_pos:
                    self._statuses[i] = 2
                    return self._rewards["dropoff_success"]
                else:
                    return self._rewards["illegal_action"]
        return self._rewards["illegal_action"]

    # ------------------------------------------------------------------
    # State encoding / decoding
    # ------------------------------------------------------------------

    def _encode(self) -> int:
        G, P, L = self.grid_size, self.n_passengers, self.n_locations
        LP  = L ** P
        SP  = 3 ** P

        taxi_pos = self._taxi_row * G + self._taxi_col

        stat_enc = 0
        for s in self._statuses:
            stat_enc = stat_enc * 3 + s

        dest_enc = 0
        for d in self._dests:
            dest_enc = dest_enc * L + d

        return taxi_pos * (SP * LP) + stat_enc * LP + dest_enc

    def decode_state(self, state: int) -> dict:
        """Decode integer state into a structured dict.

        Returns:
            {
                "taxi_row": int,
                "taxi_col": int,
                "grid_size": int,
                "locations": list[tuple],
                "passengers": [
                    {"status": 0/1/2, "pickup_loc": int, "dest": int},
                    ...
                ],
            }
        """
        G, P, L = self.grid_size, self.n_passengers, self.n_locations
        LP = L ** P
        SP = 3 ** P

        dest_enc  = state % LP;  state //= LP
        stat_enc  = state % SP;  state //= SP
        taxi_pos  = state

        taxi_row  = taxi_pos // G
        taxi_col  = taxi_pos % G

        # Decode statuses (big-endian base-3)
        statuses: list[int] = []
        tmp = stat_enc
        for _ in range(P):
            statuses.append(tmp % 3)
            tmp //= 3
        statuses.reverse()

        # Decode destinations (big-endian base-L)
        dests: list[int] = []
        tmp = dest_enc
        for _ in range(P):
            dests.append(tmp % L)
            tmp //= L
        dests.reverse()

        return {
            "taxi_row":  int(taxi_row),
            "taxi_col":  int(taxi_col),
            "grid_size": G,
            "locations": self.locations,
            "passengers": [
                {
                    "status":     statuses[i],
                    "pickup_loc": i,
                    "dest":       dests[i],
                }
                for i in range(P)
            ],
        }

    def get_state_features(self, state: int) -> np.ndarray:
        """Compact feature vector for DQN (size = n_state_features)."""
        d = self.decode_state(state)
        G, L = self.grid_size, self.n_locations
        feats: list[float] = [
            d["taxi_row"] / max(1, G - 1),
            d["taxi_col"] / max(1, G - 1),
        ]
        for p in d["passengers"]:
            oh_status = [0.0, 0.0, 0.0]
            oh_status[p["status"]] = 1.0
            feats.extend(oh_status)
            oh_dest = [0.0] * L
            oh_dest[p["dest"]] = 1.0
            feats.extend(oh_dest)
        return np.array(feats, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium-compatible property stubs
    # ------------------------------------------------------------------

    @property
    def action_space(self):
        class _S:
            def __init__(self, n): self.n = n
        return _S(self.n_actions)

    @property
    def observation_space(self):
        class _S:
            def __init__(self, n): self.n = n
        return _S(self.n_states)

    def __repr__(self) -> str:
        G, P = self.grid_size, self.n_passengers
        return (
            f"CustomTaxiEnv(grille={G}×{G}, passagers={P}, "
            f"états={self.n_states:,}, features={self.n_state_features})"
        )

# gui.py — Interface graphique Tkinter pour TaxiDriver

from __future__ import annotations

import os
import queue
import random
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

# Ensure src/ is in path
sys.path.insert(0, os.path.dirname(__file__))

from agents.brute_force import BruteForceAgent
from agents.dqn import DQNAgent
from agents.q_learning import QLearningAgent
from config import OPTIMIZED_PARAMS
from environment.taxi_wrapper import TaxiWrapper
from utils.metrics import MetricsTracker

# ── Grid constants ───────────────────────────────────────────────────────────
CELL_SIZE = 70
GRID_ROWS = 5
GRID_COLS = 5

# Location positions: index -> (row, col)
LOCATIONS = {0: (0, 0), 1: (0, 4), 2: (4, 0), 3: (4, 3)}
LOCATION_NAMES = {0: "R", 1: "G", 2: "Y", 3: "B"}
LOCATION_COLORS = {"R": "#FF6666", "G": "#66DD66", "Y": "#DDDD44", "B": "#6699FF"}

# Walls: (row, col_left, col_right) — vertical wall between col_left and col_right
WALLS = [
    (0, 1, 2),
    (1, 1, 2),
    (3, 0, 1),
    (3, 2, 3),
    (4, 0, 1),
    (4, 2, 3),
]

ACTION_NAMES = ["Sud", "Nord", "Est", "Ouest", "Pickup", "Dropoff"]


# ── Stream redirect ──────────────────────────────────────────────────────────
class _StreamToQueue:
    """Redirect write() calls to a queue as ("log", text) messages."""

    def __init__(self, q: queue.Queue) -> None:
        self._q = q

    def write(self, text: str) -> None:
        if text and text.strip():
            self._q.put(("log", text))

    def flush(self) -> None:
        pass


# ── Main GUI class ───────────────────────────────────────────────────────────
class TaxiDriverGUI:
    """Tkinter interface for the TaxiDriver reinforcement learning project."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Taxi Driver — Reinforcement Learning  |  T-AIA-902")
        self.root.configure(bg="#1e1e1e")
        self.root.minsize(1100, 720)

        self._q: queue.Queue = queue.Queue()
        self._running = False
        self._training_thread: threading.Thread | None = None

        self._build_ui()
        self._poll_queue()

    # ====================================================================
    # UI Construction
    # ====================================================================

    def _build_ui(self) -> None:
        # ── Title bar ──
        title_frame = tk.Frame(self.root, bg="#252526", height=48)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        tk.Label(
            title_frame,
            text="  TAXI DRIVER — Reinforcement Learning",
            bg="#252526", fg="#ffffff",
            font=("Segoe UI", 13, "bold"),
        ).pack(side=tk.LEFT, padx=16, pady=10)

        tk.Label(
            title_frame,
            text="T-AIA-902  ",
            bg="#252526", fg="#888888",
            font=("Segoe UI", 11),
        ).pack(side=tk.RIGHT, pady=10)

        # ── Body ──
        body = tk.Frame(self.root, bg="#1e1e1e")
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = tk.Frame(body, bg="#252526", width=315)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        right = tk.Frame(body, bg="#1e1e1e")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_config(left)
        self._build_output(right)

    # ── Config panel ─────────────────────────────────────────────────────
    def _build_config(self, parent: tk.Frame) -> None:
        tk.Label(
            parent, text="Configuration",
            bg="#252526", fg="#cccccc",
            font=("Segoe UI", 12, "bold"),
        ).pack(pady=(12, 4))

        # Scrollable area
        canvas = tk.Canvas(parent, bg="#252526", highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg="#252526")
        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=inner, anchor="nw", width=310)
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._build_mode_section(inner)
        self._build_algo_section(inner)
        self._build_reward_section(inner)
        self._build_hyper_section(inner)
        self._build_episodes_section(inner)
        self._build_buttons(inner)

    def _section(self, parent: tk.Frame, title: str) -> None:
        tk.Label(
            parent, text=title,
            bg="#252526", fg="#9cdcfe",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", padx=14, pady=(12, 0))
        ttk.Separator(parent).pack(fill=tk.X, padx=14, pady=(2, 5))

    def _radio(self, parent, text, variable, value, command=None) -> tk.Radiobutton:
        rb = tk.Radiobutton(
            parent, text=text,
            variable=variable, value=value,
            bg="#252526", fg="#cccccc",
            selectcolor="#3a3d41",
            activebackground="#252526", activeforeground="#ffffff",
            font=("Segoe UI", 9),
            command=command,
        )
        rb.pack(anchor="w", padx=28, pady=1)
        return rb

    def _entry_row(self, parent, label: str, var: tk.StringVar, width: int = 9) -> None:
        row = tk.Frame(parent, bg="#252526")
        row.pack(fill=tk.X, padx=28, pady=1)
        tk.Label(
            row, text=label,
            bg="#252526", fg="#aaaaaa",
            font=("Segoe UI", 9), width=17, anchor="w",
        ).pack(side=tk.LEFT)
        tk.Entry(
            row, textvariable=var, width=width,
            bg="#3c3f41", fg="#ffffff", insertbackground="white",
            relief=tk.FLAT, font=("Segoe UI", 9),
        ).pack(side=tk.LEFT)

    def _build_mode_section(self, parent: tk.Frame) -> None:
        self._section(parent, "MODE")
        self._mode_var = tk.IntVar(value=1)
        self._radio(parent, "User Mode (configuration manuelle)",
                    self._mode_var, 1, self._on_mode_change)
        self._radio(parent, "Time-Limited Mode (paramètres optimisés)",
                    self._mode_var, 2, self._on_mode_change)

    def _build_algo_section(self, parent: tk.Frame) -> None:
        self._section(parent, "ALGORITHME")
        self._algo_var = tk.IntVar(value=2)
        self._bf_radio = self._radio(
            parent, "Brute Force (aléatoire)", self._algo_var, 1, self._on_algo_change)
        self._radio(parent, "Q-Learning (tabulaire)", self._algo_var, 2, self._on_algo_change)
        self._radio(parent, "DQN (réseau de neurones)", self._algo_var, 3, self._on_algo_change)

    def _build_reward_section(self, parent: tk.Frame) -> None:
        self._section(parent, "REWARD SHAPING")
        self._reward_var = tk.StringVar(value="default")
        for m in ("default", "distance", "custom"):
            self._radio(parent, m, self._reward_var, m, self._on_reward_change)

        self._custom_frame = tk.Frame(parent, bg="#252526")
        self._custom_step_var = tk.StringVar(value="-1.0")
        self._custom_dropoff_var = tk.StringVar(value="20.0")
        self._custom_illegal_var = tk.StringVar(value="-10.0")
        self._entry_row(self._custom_frame, "Pénalité step:", self._custom_step_var)
        self._entry_row(self._custom_frame, "Bonus dropoff:", self._custom_dropoff_var)
        self._entry_row(self._custom_frame, "Pénalité illégale:", self._custom_illegal_var)

    def _build_hyper_section(self, parent: tk.Frame) -> None:
        self._section(parent, "HYPERPARAMÈTRES")
        self._hyper_outer = tk.Frame(parent, bg="#252526")
        self._hyper_outer.pack(fill=tk.X)

        # Q-Learning params
        self._ql_frame = tk.Frame(self._hyper_outer, bg="#252526")
        self._alpha_var = tk.StringVar(value="0.8")
        self._gamma_var = tk.StringVar(value="0.99")
        self._eps_var = tk.StringVar(value="1.0")
        self._eps_decay_var = tk.StringVar(value="0.999")
        self._eps_min_var = tk.StringVar(value="0.01")
        for label, var in [
            ("Alpha (lr):", self._alpha_var),
            ("Gamma:", self._gamma_var),
            ("Epsilon init:", self._eps_var),
            ("Epsilon decay:", self._eps_decay_var),
            ("Epsilon min:", self._eps_min_var),
        ]:
            self._entry_row(self._ql_frame, label, var)

        # DQN params
        self._dqn_frame = tk.Frame(self._hyper_outer, bg="#252526")
        self._lr_var = tk.StringVar(value="0.001")
        self._dqn_gamma_var = tk.StringVar(value="0.99")
        self._dqn_eps_var = tk.StringVar(value="1.0")
        self._dqn_eps_decay_var = tk.StringVar(value="0.995")
        self._dqn_eps_min_var = tk.StringVar(value="0.01")
        self._batch_var = tk.StringVar(value="64")
        self._target_var = tk.StringVar(value="100")
        for label, var in [
            ("Learning rate:", self._lr_var),
            ("Gamma:", self._dqn_gamma_var),
            ("Epsilon init:", self._dqn_eps_var),
            ("Epsilon decay:", self._dqn_eps_decay_var),
            ("Epsilon min:", self._dqn_eps_min_var),
            ("Batch size:", self._batch_var),
            ("Target update:", self._target_var),
        ]:
            self._entry_row(self._dqn_frame, label, var)

        self._ql_frame.pack(fill=tk.X)  # Q-Learning shown by default

    def _build_episodes_section(self, parent: tk.Frame) -> None:
        self._section(parent, "ÉPISODES / DURÉE")
        self._ep_outer = tk.Frame(parent, bg="#252526")
        self._ep_outer.pack(fill=tk.X)

        self._user_ep_frame = tk.Frame(self._ep_outer, bg="#252526")
        self._n_train_var = tk.StringVar(value="5000")
        self._entry_row(self._user_ep_frame, "Entraînement:", self._n_train_var)
        self._user_ep_frame.pack(fill=tk.X)

        self._time_frame = tk.Frame(self._ep_outer, bg="#252526")
        self._time_var = tk.StringVar(value="60")
        self._entry_row(self._time_frame, "Durée (sec):", self._time_var)

        self._n_test_var = tk.StringVar(value="100")
        self._entry_row(parent, "Épisodes test:", self._n_test_var)

    def _build_buttons(self, parent: tk.Frame) -> None:
        btns = tk.Frame(parent, bg="#252526")
        btns.pack(fill=tk.X, padx=14, pady=16)

        self._start_btn = tk.Button(
            btns, text="▶  Démarrer",
            bg="#4CAF50", fg="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, padx=10, pady=8,
            cursor="hand2",
            command=self._start,
        )
        self._start_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))

        self._stop_btn = tk.Button(
            btns, text="■  Arrêter",
            bg="#f44336", fg="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, padx=10, pady=8,
            cursor="hand2",
            state=tk.DISABLED,
            command=self._stop,
        )
        self._stop_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

    # ── Output panel ──────────────────────────────────────────────────────
    def _build_output(self, parent: tk.Frame) -> None:
        # Top row: grid + metrics
        top = tk.Frame(parent, bg="#1e1e1e")
        top.pack(fill=tk.X, pady=(0, 8))

        # Grid canvas
        grid_wrap = tk.Frame(top, bg="#1e1e1e")
        grid_wrap.pack(side=tk.LEFT, padx=(0, 16))
        tk.Label(
            grid_wrap, text="Grille Taxi-v3",
            bg="#1e1e1e", fg="#888888",
            font=("Segoe UI", 9),
        ).pack()
        cs = CELL_SIZE * GRID_COLS
        self._canvas = tk.Canvas(
            grid_wrap, width=cs, height=cs,
            bg="#111111", highlightthickness=1, highlightbackground="#444444",
        )
        self._canvas.pack()
        self._draw_empty_grid()

        # Metrics panel
        m_frame = tk.Frame(top, bg="#252526", padx=14, pady=10)
        m_frame.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(
            m_frame, text="Métriques",
            bg="#252526", fg="#9cdcfe",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        self._m_vars: dict[str, tk.StringVar] = {}
        for key, label in [
            ("status",      "Statut"),
            ("episodes",    "Épisodes train"),
            ("success_rate","Taux succès"),
            ("mean_steps",  "Steps moyens"),
            ("mean_reward", "Récompense moy"),
            ("epsilon",     "Epsilon final"),
        ]:
            row = tk.Frame(m_frame, bg="#252526")
            row.pack(fill=tk.X, pady=2)
            tk.Label(
                row, text=f"{label}:",
                bg="#252526", fg="#888888",
                font=("Segoe UI", 9), width=17, anchor="w",
            ).pack(side=tk.LEFT)
            sv = tk.StringVar(value="—")
            tk.Label(
                row, textvariable=sv,
                bg="#252526", fg="#ffffff",
                font=("Segoe UI", 9, "bold"),
            ).pack(side=tk.LEFT)
            self._m_vars[key] = sv

        # Progress bar
        pb_frame = tk.Frame(parent, bg="#1e1e1e")
        pb_frame.pack(fill=tk.X, pady=(0, 6))
        tk.Label(
            pb_frame, text="Progression :",
            bg="#1e1e1e", fg="#888888",
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, padx=(0, 8))
        self._pb_var = tk.DoubleVar(value=0)
        ttk.Progressbar(pb_frame, variable=self._pb_var, maximum=100).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        self._pb_lbl = tk.Label(
            pb_frame, text="0 %",
            bg="#1e1e1e", fg="#888888",
            font=("Segoe UI", 9), width=5,
        )
        self._pb_lbl.pack(side=tk.LEFT, padx=(6, 0))

        # Log area
        tk.Label(
            parent, text="Journal d'exécution :",
            bg="#1e1e1e", fg="#888888",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(4, 2))
        self._log = scrolledtext.ScrolledText(
            parent,
            bg="#111111", fg="#cccccc",
            font=("Consolas", 9),
            state=tk.DISABLED, wrap=tk.WORD, height=16,
        )
        self._log.pack(fill=tk.BOTH, expand=True)
        self._log.tag_configure("header",  foreground="#FFC107", font=("Consolas", 9, "bold"))
        self._log.tag_configure("success", foreground="#4CAF50")
        self._log.tag_configure("error",   foreground="#f44336")
        self._log.tag_configure("info",    foreground="#2196F3")

    # ====================================================================
    # Grid Drawing
    # ====================================================================

    def _draw_empty_grid(self) -> None:
        self._canvas.delete("all")
        cs = CELL_SIZE
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                self._canvas.create_rectangle(
                    c*cs, r*cs, c*cs+cs, r*cs+cs,
                    fill="#1a1a2e", outline="#333333",
                )
        for loc_idx, (r, c) in LOCATIONS.items():
            name = LOCATION_NAMES[loc_idx]
            self._canvas.create_text(
                c*cs + cs//2, r*cs + cs//2,
                text=name, fill=LOCATION_COLORS[name],
                font=("Segoe UI", 20, "bold"),
            )
        self._draw_walls()

    def _draw_state(
        self,
        state_dict: dict,
        action: int | None = None,
        step: int = 0,
        reward: float = 0,
        total: float = 0,
    ) -> None:
        self._canvas.delete("all")
        cs = CELL_SIZE
        taxi_r   = state_dict["taxi_row"]
        taxi_c   = state_dict["taxi_col"]
        pass_loc = state_dict["passenger_loc"]
        dest     = state_dict["destination"]
        in_taxi  = pass_loc == 4
        dest_pos = LOCATIONS[dest]

        # Cells
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                bg = "#2a1a0a" if (r, c) == dest_pos else "#1a1a2e"
                self._canvas.create_rectangle(
                    c*cs, r*cs, c*cs+cs, r*cs+cs,
                    fill=bg, outline="#333333",
                )

        # Location labels
        for loc_idx, (r, c) in LOCATIONS.items():
            name = LOCATION_NAMES[loc_idx]
            x, y = c*cs + cs//2, r*cs + cs//2
            color = "#FFD700" if loc_idx == dest else LOCATION_COLORS[name]
            if loc_idx == pass_loc and not in_taxi:
                # Passenger waiting here
                self._canvas.create_oval(
                    x-14, y-14, x+14, y+14,
                    fill="#1e3a6e", outline="#4499FF", width=2,
                )
                self._canvas.create_text(x, y, text=name, fill="#ffffff",
                                         font=("Segoe UI", 12, "bold"))
            else:
                self._canvas.create_text(x, y, text=name, fill=color,
                                         font=("Segoe UI", 20, "bold"))

        # Taxi
        margin = 10
        tx0 = taxi_c*cs + margin
        ty0 = taxi_r*cs + margin
        tx1 = taxi_c*cs + cs - margin
        ty1 = taxi_r*cs + cs - margin
        taxi_color = "#44CC44" if in_taxi else "#FFD700"
        self._canvas.create_rectangle(
            tx0, ty0, tx1, ty1,
            fill=taxi_color, outline="#888888", width=2,
        )
        taxi_label = "T+" if in_taxi else "T"
        self._canvas.create_text(
            taxi_c*cs + cs//2, taxi_r*cs + cs//2,
            text=taxi_label, fill="#1a1a1a",
            font=("Segoe UI", 14, "bold"),
        )

        self._draw_walls()

        # Step info overlay at bottom
        if action is not None:
            info = (
                f"Step {step}  ·  {ACTION_NAMES[action]}"
                f"  ·  r={reward:+.0f}  ·  total={total:+.0f}"
            )
            self._canvas.create_text(
                CELL_SIZE*GRID_COLS//2, CELL_SIZE*GRID_ROWS - 4,
                text=info, fill="#aaaaaa",
                font=("Segoe UI", 8), anchor="s",
            )

    def _draw_walls(self) -> None:
        cs = CELL_SIZE
        for (r, cl, cr) in WALLS:
            x = cr * cs
            self._canvas.create_line(
                x, r*cs, x, (r+1)*cs,
                fill="#888888", width=3,
            )

    # ====================================================================
    # Event Handlers
    # ====================================================================

    def _on_mode_change(self) -> None:
        mode = self._mode_var.get()
        if mode == 1:
            self._bf_radio.configure(state=tk.NORMAL)
            self._time_frame.pack_forget()
            self._user_ep_frame.pack(fill=tk.X)
        else:
            if self._algo_var.get() == 1:
                self._algo_var.set(2)
                self._on_algo_change()
            self._bf_radio.configure(state=tk.DISABLED)
            self._user_ep_frame.pack_forget()
            self._time_frame.pack(fill=tk.X)

    def _on_algo_change(self) -> None:
        self._ql_frame.pack_forget()
        self._dqn_frame.pack_forget()
        algo = self._algo_var.get()
        if algo == 2:
            self._ql_frame.pack(fill=tk.X)
        elif algo == 3:
            self._dqn_frame.pack(fill=tk.X)

    def _on_reward_change(self) -> None:
        if self._reward_var.get() == "custom":
            self._custom_frame.pack(fill=tk.X, padx=14, pady=4)
        else:
            self._custom_frame.pack_forget()

    # ====================================================================
    # Training Control
    # ====================================================================

    def _start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_btn.configure(state=tk.DISABLED)
        self._stop_btn.configure(state=tk.NORMAL)
        self._pb_var.set(0)
        self._pb_lbl.configure(text="0 %")
        self._clear_log()
        for sv in self._m_vars.values():
            sv.set("—")
        self._m_vars["status"].set("En cours…")
        self._draw_empty_grid()

        self._training_thread = threading.Thread(target=self._worker, daemon=True)
        self._training_thread.start()

    def _stop(self) -> None:
        self._running = False
        self._log_line("Arrêt demandé…", "error")

    # ── Background worker ─────────────────────────────────────────────────
    def _worker(self) -> None:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = _StreamToQueue(self._q)
        try:
            if self._mode_var.get() == 1:
                self._worker_user_mode()
            else:
                self._worker_time_limited()
        except Exception as exc:
            import traceback
            self._q.put(("log", f"\nErreur : {exc}\n{traceback.format_exc()}"))
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            self._q.put(("done", None))

    def _worker_user_mode(self) -> None:
        algo        = self._algo_var.get()
        reward_mode = self._reward_var.get()
        n_test      = int(self._n_test_var.get())

        custom_rewards = None
        if reward_mode == "custom":
            custom_rewards = {
                "step":            float(self._custom_step_var.get()),
                "dropoff_success": float(self._custom_dropoff_var.get()),
                "illegal_action":  float(self._custom_illegal_var.get()),
            }

        env = TaxiWrapper(reward_mode=reward_mode, custom_rewards=custom_rewards)
        algo_name = {1: "Brute Force", 2: "Q-Learning", 3: "DQN"}[algo]
        print(f"\n{'='*50}")
        print(f"  USER MODE  |  {algo_name}")
        print(f"{'='*50}\n")

        # ── Brute Force ──
        if algo == 1:
            agent = BruteForceAgent(env)
            print(f"[BruteForce] Test sur {n_test} épisodes…")
            test_tracker = self._run_test(agent, n_test, env)
            self._send_final_metrics(test_tracker, n_train=0, epsilon=None)
            self._queue_replay(agent, env, n=3)
            from utils.visualization import plot_steps_distribution, plot_summary_table
            plot_steps_distribution({"BruteForce": test_tracker})
            plot_summary_table({"BruteForce": test_tracker.summary()})
            env.close()
            return

        n_train = int(self._n_train_var.get())

        # ── Q-Learning ──
        if algo == 2:
            agent = QLearningAgent(
                env,
                alpha=float(self._alpha_var.get()),
                gamma=float(self._gamma_var.get()),
                epsilon=float(self._eps_var.get()),
                epsilon_decay=float(self._eps_decay_var.get()),
                epsilon_min=float(self._eps_min_var.get()),
            )
            label = "Q-Learning (user)"
        # ── DQN ──
        else:
            agent = DQNAgent(
                env,
                lr=float(self._lr_var.get()),
                gamma=float(self._dqn_gamma_var.get()),
                epsilon=float(self._dqn_eps_var.get()),
                epsilon_decay=float(self._dqn_eps_decay_var.get()),
                epsilon_min=float(self._dqn_eps_min_var.get()),
                batch_size=int(self._batch_var.get()),
                target_update_freq=int(self._target_var.get()),
            )
            label = "DQN (user)"

        print(repr(agent))
        train_tracker = self._run_train(agent, n_train)

        if not self._running:
            env.close()
            return

        print(f"\nTest sur {n_test} épisodes…")
        test_tracker = self._run_test(agent, n_test, env)
        self._send_final_metrics(test_tracker, n_train=n_train,
                                 epsilon=getattr(agent, "epsilon", None))
        self._queue_replay(agent, env, n=3)

        print("\nGénération des graphiques…")
        from utils.visualization import (
            plot_epsilon_decay, plot_learning_curves,
            plot_steps_distribution, plot_steps_evolution, plot_summary_table,
        )
        plot_learning_curves({label: train_tracker})
        plot_steps_evolution({label: train_tracker})
        if hasattr(agent, "epsilon_history") and agent.epsilon_history:
            plot_epsilon_decay({label: agent.epsilon_history})
        plot_steps_distribution({label: test_tracker})
        plot_summary_table({label: test_tracker.summary()})

        env.close()

    def _worker_time_limited(self) -> None:
        algo        = self._algo_var.get()
        time_budget = int(self._time_var.get())
        n_test      = int(self._n_test_var.get())

        env = TaxiWrapper(reward_mode="default")
        algo_name = {2: "Q-Learning", 3: "DQN"}.get(algo, "Q-Learning")
        print(f"\n{'='*50}")
        print(f"  TIME-LIMITED MODE  |  {algo_name} (paramètres optimisés)")
        print(f"{'='*50}\n")

        if algo == 3:
            agent = DQNAgent(env, **OPTIMIZED_PARAMS["dqn"])
            label = "DQN (tuned)"
        else:
            agent = QLearningAgent(env, **OPTIMIZED_PARAMS["ql"])
            label = "Q-Learning (tuned)"

        print(repr(agent))
        print(f"Entraînement pendant {time_budget}s…\n")

        tracker   = MetricsTracker(n_episodes=999_999, log_interval=200, phase="Training")
        ep        = 0
        deadline  = time.perf_counter() + time_budget
        last_pct  = -1

        while time.perf_counter() < deadline and self._running:
            state, _ = env.reset()
            total_reward = 0.0
            steps = 0
            terminated = truncated = False
            tracker.begin_episode(ep)

            while not (terminated or truncated):
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)

                if isinstance(agent, QLearningAgent):
                    agent.update(state, action, reward, next_state, terminated or truncated)
                else:
                    agent.buffer.push(state, action, reward, next_state, terminated or truncated)
                    loss = agent.learn()
                    if loss is not None:
                        agent.loss_history.append(loss)
                    agent._step_count += 1
                    if agent._step_count % agent.target_update_freq == 0:
                        agent._sync_target()

                state = next_state
                total_reward += reward
                steps += 1

            agent._decay_epsilon()
            tracker.end_episode(ep, steps, total_reward, terminated)
            ep += 1

            elapsed = time.perf_counter() - (deadline - time_budget)
            pct = min(100, int(elapsed / time_budget * 100))
            if pct != last_pct and pct % 2 == 0:
                last_pct = pct
                self._q.put(("progress", pct))
                if tracker.records:
                    s = tracker.summary()
                    self._q.put(("metrics_partial", {
                        "episodes":    f"{ep} épisodes",
                        "success_rate": f"{s['success_rate']:.1f} %",
                        "mean_steps":  f"{s['mean_steps']:.1f}",
                        "mean_reward": f"{s['mean_reward']:.2f}",
                        "epsilon":     f"{agent.epsilon:.4f}",
                    }))

        elapsed = time.perf_counter() - (deadline - time_budget)
        print(f"Entraînement terminé — {ep} épisodes en {elapsed:.1f}s\n")

        if not self._running:
            env.close()
            return

        print(f"Test sur {n_test} épisodes…")
        test_tracker = self._run_test(agent, n_test, env)
        self._send_final_metrics(test_tracker, n_train=ep,
                                 epsilon=getattr(agent, "epsilon", None))
        self._queue_replay(agent, env, n=3)

        from utils.visualization import (
            plot_epsilon_decay, plot_learning_curves,
            plot_steps_distribution, plot_steps_evolution, plot_summary_table,
        )
        plot_learning_curves({label: tracker})
        plot_steps_evolution({label: tracker})
        if hasattr(agent, "epsilon_history") and agent.epsilon_history:
            plot_epsilon_decay({label: agent.epsilon_history})
        plot_steps_distribution({label: test_tracker})
        plot_summary_table({label: test_tracker.summary()})

        env.close()

    # ── Training / test loops ─────────────────────────────────────────────
    def _run_train(self, agent, n_episodes: int) -> MetricsTracker:
        """Custom training loop with periodic progress reporting to the queue."""
        log_interval = max(1, n_episodes // 10)
        tracker = MetricsTracker(
            n_episodes=n_episodes, log_interval=log_interval, phase="Training"
        )
        report_every = max(1, n_episodes // 100)

        for ep in range(n_episodes):
            if not self._running:
                break

            state, _ = agent.env.reset()
            total_reward = 0.0
            steps = 0
            terminated = truncated = False
            tracker.begin_episode(ep)

            while not (terminated or truncated):
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = agent.env.step(action)

                if isinstance(agent, QLearningAgent):
                    agent.update(state, action, reward, next_state, terminated or truncated)
                else:
                    agent.buffer.push(state, action, reward, next_state, terminated or truncated)
                    loss = agent.learn()
                    if loss is not None:
                        agent.loss_history.append(loss)
                    agent._step_count += 1
                    if agent._step_count % agent.target_update_freq == 0:
                        agent._sync_target()

                state = next_state
                total_reward += reward
                steps += 1

            agent._decay_epsilon()
            tracker.end_episode(ep, steps, total_reward, terminated)

            # Progress report
            if (ep + 1) % report_every == 0:
                pct = (ep + 1) / n_episodes * 100
                self._q.put(("progress", pct))
                s = tracker.summary()
                self._q.put(("metrics_partial", {
                    "episodes":    f"{ep+1}/{n_episodes}",
                    "success_rate": f"{s['success_rate']:.1f} %",
                    "mean_steps":  f"{s['mean_steps']:.1f}",
                    "mean_reward": f"{s['mean_reward']:.2f}",
                    "epsilon":     f"{agent.epsilon:.4f}" if hasattr(agent, "epsilon") else "N/A",
                }))
            elif (ep + 1) % log_interval == 0:
                tracker._log(ep)

        return tracker

    def _run_test(self, agent, n_episodes: int, env: TaxiWrapper) -> MetricsTracker:
        """Greedy test loop."""
        log_interval = max(1, n_episodes // 5)
        tracker = MetricsTracker(
            n_episodes=n_episodes, log_interval=log_interval, phase="Test"
        )

        for ep in range(n_episodes):
            if not self._running:
                break
            state, _ = env.reset()
            total_reward = 0.0
            steps = 0
            terminated = truncated = False
            tracker.begin_episode(ep)

            while not (terminated or truncated):
                action = agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                state = next_state
                total_reward += reward
                steps += 1

            tracker.end_episode(ep, steps, total_reward, terminated)

        tracker.print_summary()
        return tracker

    # ── Replay helpers ────────────────────────────────────────────────────
    def _queue_replay(self, agent, env: TaxiWrapper, n: int = 3) -> None:
        """Collect greedy episode trajectories, send to queue for animation."""
        print(f"\nCollecte de {n} replays pour visualisation…")
        all_episodes: list[list[dict]] = []

        for _ in range(n):
            seed = random.randint(0, 9999)
            replay_env = TaxiWrapper(reward_mode=env.reward_mode)
            state, _ = replay_env.reset(seed=seed)

            steps: list[dict] = [
                {"state_dict": replay_env.decode_state(state),
                 "action": None, "step": 0, "reward": 0, "total": 0}
            ]
            total_reward = 0.0
            step_n = 0
            terminated = truncated = False

            while not (terminated or truncated):
                action = agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = replay_env.step(action)
                total_reward += reward
                step_n += 1
                steps.append({
                    "state_dict": replay_env.decode_state(next_state),
                    "action":     action,
                    "step":       step_n,
                    "reward":     reward,
                    "total":      total_reward,
                })
                state = next_state

            all_episodes.append(steps)
            replay_env.close()

        self._q.put(("replay_ready", all_episodes))

    def _send_final_metrics(
        self, test_tracker: MetricsTracker, n_train: int, epsilon
    ) -> None:
        s = test_tracker.summary()
        self._q.put(("metrics_final", {
            "episodes":    str(n_train),
            "success_rate": f"{s.get('success_rate', 0):.1f} %",
            "mean_steps":  f"{s.get('mean_steps', 0):.1f}",
            "mean_reward": f"{s.get('mean_reward', 0):.2f}",
            "epsilon":     f"{epsilon:.4f}" if epsilon is not None else "N/A",
        }))

    # ====================================================================
    # Queue Polling
    # ====================================================================

    def _poll_queue(self) -> None:
        try:
            while True:
                msg_type, payload = self._q.get_nowait()
                if msg_type == "log":
                    self._log_line(payload)
                elif msg_type == "progress":
                    self._pb_var.set(payload)
                    self._pb_lbl.configure(text=f"{payload:.0f} %")
                elif msg_type in ("metrics_partial", "metrics_final"):
                    for k, v in payload.items():
                        if k in self._m_vars:
                            self._m_vars[k].set(v)
                elif msg_type == "replay_ready":
                    self._log_line("\nAnimation des replays…", "info")
                    self._animate_replays(payload, ep_idx=0, step_idx=0)
                elif msg_type == "done":
                    self._running = False
                    self._start_btn.configure(state=tk.NORMAL)
                    self._stop_btn.configure(state=tk.DISABLED)
                    self._pb_var.set(100)
                    self._pb_lbl.configure(text="100 %")
                    self._m_vars["status"].set("Terminé")
                    self._log_line("\nSession terminée.", "success")
        except queue.Empty:
            pass
        self.root.after(50, self._poll_queue)

    # ====================================================================
    # Replay Animation
    # ====================================================================

    def _animate_replays(
        self, all_episodes: list[list[dict]], ep_idx: int, step_idx: int
    ) -> None:
        if ep_idx >= len(all_episodes):
            return

        episode = all_episodes[ep_idx]
        if step_idx >= len(episode):
            # Pause between episodes, then advance
            self.root.after(
                900,
                lambda: self._animate_replays(all_episodes, ep_idx + 1, 0),
            )
            return

        step = episode[step_idx]
        self._draw_state(
            step["state_dict"],
            step["action"],
            step["step"],
            step["reward"],
            step["total"],
        )
        self.root.after(
            320,
            lambda: self._animate_replays(all_episodes, ep_idx, step_idx + 1),
        )

    # ====================================================================
    # Log Helpers
    # ====================================================================

    def _clear_log(self) -> None:
        self._log.configure(state=tk.NORMAL)
        self._log.delete(1.0, tk.END)
        self._log.configure(state=tk.DISABLED)

    def _log_line(self, text: str, tag: str | None = None) -> None:
        self._log.configure(state=tk.NORMAL)
        self._log.insert(tk.END, text.rstrip() + "\n", tag or "")
        self._log.see(tk.END)
        self._log.configure(state=tk.DISABLED)

    # ====================================================================
    # Run
    # ====================================================================

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = TaxiDriverGUI()
    app.run()


if __name__ == "__main__":
    main()

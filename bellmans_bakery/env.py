import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import gymnasium as gym
import numpy as np


# ------------------------------
# Domain configuration constants
# ------------------------------

ITEM_NAMES = [
    "mini_red_velvet",
    "raspberry_matcha_roll",
    "strawberry_cream_slice",
    "chocolate_almond_drip_cake",
    "chocolate_orange_roll",
]

# Size units fit into 4u oven capacity
ITEM_SIZE = np.array([3.0, 1.5, 1.0, 4.0, 1.5], dtype=np.float32)

# Bake times in seconds (1 tick = 10s)
ITEM_BAKE_SECONDS = np.array([90, 60, 36, 120, 60], dtype=np.float32)

# Batch yields per completed bake
ITEM_YIELD = np.array([2, 4, 4, 1, 4], dtype=np.int32)

# Prices and costs
ITEM_PRICE = np.array([6.00, 4.50, 3.50, 8.00, 4.50], dtype=np.float32)
ITEM_COST = 0.40 * ITEM_PRICE

# Demand mix (probabilities, skewed)
DEMAND_MIX = np.array([0.15, 0.20, 0.30, 0.15, 0.20], dtype=np.float32)


@dataclass
class Customer:
    """Queue element; minimal state needed for control decisions."""

    desired_item: int
    patience_ticks_remaining: int


class BellmansBakeryEnv(gym.Env):
    """
    Bellman's Bakery environment.

    One episode is a single day with discrete ticks (10 seconds per tick).
    Actions are single-step choices (serve/bake/idle); serving takes the whole
    tick. Ovens allow concurrent baking subject to capacity constraints.

    Observation encodes time-of-day features, inventory, coarse oven status,
    queue summary (first K customers), queue length, and the daily price
    multiplier. An action mask is provided in `info["action_mask"]`.
    """

    metadata = {"render_modes": ["human"], "render_fps": 15}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        super().__init__()
        self.cfg = config or {}

        # Core time settings
        self.seconds_per_tick = 10
        self.day_ticks = int(self.cfg.get("day_ticks", 240))

        # Queuing and visibility
        self.queue_cap = int(self.cfg.get("queue_cap", 12))
        self.observe_k_customers = int(self.cfg.get("observe_k", 5))

        # Ovens (capacity model)
        self.num_ovens = int(self.cfg.get("num_ovens", 2))
        self.oven_capacity = float(self.cfg.get("oven_capacity", 4.0))
        # How many customers can be served within a single tick (raise to ease bottleneck)
        self.serve_per_tick = int(self.cfg.get("serve_per_tick", 3))

        # Demand settings
        self.avg_customers_per_day = float(self.cfg.get("avg_customers_per_day", 60))
        self.first_arrival_delay_ticks = int(
            self.cfg.get("first_arrival_delay_ticks", 2)
        )  # 20s
        self.enable_nonstationarity = bool(self.cfg.get("enable_nonstationarity", True))
        self.daily_drift_pct = float(self.cfg.get("daily_drift_pct", 0.10))
        self.weekly_item_swing_pct = float(self.cfg.get("weekly_item_swing_pct", 0.10))

        # Rewards
        self.wait_penalty_per_tick = float(self.cfg.get("wait_penalty_per_tick", 0.01))
        self.abandon_penalty = float(self.cfg.get("abandon_penalty", 0.5))
        self.serve_bonus = float(self.cfg.get("serve_bonus", 0.1))
        self.idle_penalty = float(self.cfg.get("idle_penalty", 0.0))
        self.balk_penalty = float(self.cfg.get("balk_penalty", 0.1))

        # Service time (modelled implicitly: one serve action consumes the tick)

        # Prices/costs/bandit multiplier
        self.base_prices = ITEM_PRICE.copy()
        self.costs = ITEM_COST.copy()

        # Observations: [time_sin, time_cos] + inv(5) + ovens(2) + queue_len(1)
        # + K * (item_onehot(5) + patience(1)) + price_multiplier(1)
        obs_dim = 2 + 5 + self.num_ovens + 1 + self.observe_k_customers * (5 + 1) + 1
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Actions: serve_i (5), bake_i (5), idle (1)
        self.action_meanings: List[Tuple[str, Optional[int]]] = []
        for i in range(5):
            self.action_meanings.append(("serve", i))
        for i in range(5):
            self.action_meanings.append(("bake", i))
        self.action_meanings.append(("idle", None))
        self.action_space = gym.spaces.Discrete(len(self.action_meanings))

        self.render_mode = render_mode
        self.rng = np.random.default_rng()

        # Default daily price multiplier; can be overridden via reset(options=...)
        self.daily_price_multiplier = 1.0
        # For viewer: remember the last action taken in human-readable form
        self.last_action_str = "idle"

        self._build_arrival_profile()
        self._reset_sim(day_index=0)

    # -----------------
    # Gym API
    # -----------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        day_index = (self.day_index + 1) if hasattr(self, "day_index") else 0
        # Allow external control of the daily price multiplier (e.g., bandit); default 1.0
        self.daily_price_multiplier = float(
            (options or {}).get("price_multiplier", 1.0)
        )
        self._reset_sim(day_index=day_index)
        return self._obs(), self._info()

    def step(self, action: int):
        reward = 0.0

        # Compute mask; illegal actions get a small penalty and are treated as idle
        mask = self._action_mask()
        if not mask[action]:
            reward -= 0.02  # soft guard against illegal action flailing
            action = self.action_space.n - 1  # idle

        meaning, idx = self.action_meanings[action]

        if meaning == "serve":
            self.last_action_str = f"serve {ITEM_NAMES[idx]}"
            # Serve up to 'serve_per_tick' customers this tick if possible
            for _ in range(max(1, self.serve_per_tick)):
                gained = self._attempt_serve(idx)
                if gained <= -0.01:  # no matching customer or no inventory
                    break
                reward += gained
        elif meaning == "bake":
            self.last_action_str = f"bake {ITEM_NAMES[idx]}"
            reward += self._attempt_bake(idx)
        else:  # idle
            self.last_action_str = "idle"
            reward -= self.idle_penalty

        # Advance time by one tick: progress ovens, add arrivals, update patience
        reward += self._advance_tick_effects()

        self.t += 1
        terminated = self.t >= self.day_ticks

        if terminated:
            # Waste penalty at day end
            leftover = self.inventory.copy()
            waste_cost = float(np.dot(leftover, self.costs))
            reward -= waste_cost
            # Log net profit (sales margin minus leftover cost) for fair evaluation
            self.profit -= waste_cost

        obs = self._obs()
        info = self._info()
        return obs, float(reward), bool(terminated), False, info

    def render(self):
        # Minimalistic textual render; Pygame lives in a thin renderer module later.
        inv = {ITEM_NAMES[i]: int(self.inventory[i]) for i in range(5)}
        print(
            f"t={self.t:03d}/{self.day_ticks}  profit=${self.profit:.2f}  inv={inv}  queue={len(self.queue)}"
        )

    # -----------------
    # Internal mechanics
    # -----------------

    def _reset_sim(self, day_index: int):
        self.day_index = day_index
        self.t = 0
        # Warm-start inventory (original buffer): slice=6, rolls=4 each, red velvet=2, drip=1
        # Order: [red_velvet, matcha_roll, strawberry_slice, drip_cake, chocolate_orange_roll]
        self.inventory = np.array([2, 4, 6, 1, 4], dtype=np.int32)
        self.queue: List[Customer] = []
        self.profit = 0.0
        # Evaluation counters
        self.served_count = 0
        self.abandoned_count = 0
        self.arrivals_count = 0
        self.balked_count = 0
        self.total_wait_customers_ticks = 0  # sum over time of queue length

        # Ovens: list of loads per oven; each load is (item_idx, size_u, ticks_remaining)
        self.ovens: List[List[Tuple[int, float, int]]] = [
            [] for _ in range(self.num_ovens)
        ]

        # Apply non-stationarity drift per day
        if self.enable_nonstationarity:
            self._apply_daily_drift()

        # Compute prices for the day
        self.prices_today = self.base_prices * self.daily_price_multiplier

    def _apply_daily_drift(self):
        # Daily global arrival drift around 1.0
        drift = 1.0 + self.rng.uniform(-self.daily_drift_pct, self.daily_drift_pct)
        self.arrival_profile_scaled = self.arrival_profile * drift

        # Weekly item taste swings (sinusoidal with noise)
        w = 2 * math.pi * (self.day_index % 7) / 7.0
        swings = 1.0 + self.weekly_item_swing_pct * np.array(
            [
                math.sin(w + 0.0),
                math.sin(w + 0.7),
                math.sin(w + 1.4),
                math.sin(w + 2.1),
                math.sin(w + 2.8),
            ],
            dtype=np.float32,
        )
        self.demand_mix_today = self._normalize(DEMAND_MIX * swings)

    def _build_arrival_profile(self):
        """
        Build a per-tick base arrival rate profile with morning and lunch peaks.
        The profile is later rescaled so the expected count over the day is the
        configured average.
        """
        T = self.day_ticks
        x = np.linspace(0, 1, T, dtype=np.float32)
        # Two smooth bumps: morning (~0.2) and lunch (~0.6)
        bump1 = np.exp(-0.5 * ((x - 0.25) / 0.12) ** 2)
        bump2 = np.exp(-0.5 * ((x - 0.62) / 0.10) ** 2)
        base = 0.4 * bump1 + 0.6 * bump2 + 0.2  # ensure a nonzero floor
        # Zero arrivals for the first few ticks
        base[: self.first_arrival_delay_ticks] = 0.0
        # Scale to hit the target expected arrivals
        s = np.sum(base)
        target_total = self.avg_customers_per_day
        self.arrival_profile = base * (target_total / s)
        self.arrival_profile_scaled = self.arrival_profile.copy()
        self.demand_mix_today = DEMAND_MIX.copy()

    # ------ Actions ------

    def _attempt_serve(self, item_idx: int) -> float:
        # Find earliest customer who wants this item
        pos = None
        for i, c in enumerate(self.queue):
            if c.desired_item == item_idx and self.inventory[item_idx] > 0:
                pos = i
                break
        if pos is None:
            return -0.01  # soft penalty for futile serve

        # Serve one customer; serving consumes the tick
        self.inventory[item_idx] -= 1
        price = float(self.prices_today[item_idx])
        cost = float(self.costs[item_idx])
        reward = price - cost + self.serve_bonus
        self.profit += price - cost
        self.served_count += 1
        # Remove that customer from the queue
        self.queue.pop(pos)
        return reward

    def _attempt_bake(self, item_idx: int) -> float:
        size_u = float(ITEM_SIZE[item_idx])
        # Place into the first oven with enough remaining capacity
        for oven in self.ovens:
            used = sum(load[1] for load in oven)
            if used + size_u <= self.oven_capacity:
                ticks = int(
                    math.ceil(ITEM_BAKE_SECONDS[item_idx] / self.seconds_per_tick)
                )
                oven.append((item_idx, size_u, ticks))
                # Baking cost is charged upon serving; energy cost ignored for now
                return 0.0
        # No space available
        return -0.01

    # ------ Tick advancement ------

    def _advance_tick_effects(self) -> float:
        reward = 0.0

        # Ovens progress; completed bakes add inventory
        for o in range(self.num_ovens):
            new_loads = []
            for item_idx, size_u, ticks_remaining in self.ovens[o]:
                ticks_remaining -= 1
                if ticks_remaining <= 0:
                    self.inventory[item_idx] += int(ITEM_YIELD[item_idx])
                else:
                    new_loads.append((item_idx, size_u, ticks_remaining))
            self.ovens[o] = new_loads

        # New arrivals (Poisson) with current rate
        lam = (
            float(self.arrival_profile_scaled[self.t])
            if self.t < self.day_ticks
            else 0.0
        )
        arrivals = self.rng.poisson(lam=lam)
        for _ in range(int(arrivals)):
            desired = int(self.rng.choice(len(ITEM_NAMES), p=self.demand_mix_today))
            patience_ticks = int(self.rng.integers(low=3, high=10))  # 30-90s
            self.arrivals_count += 1
            if len(self.queue) < self.queue_cap:
                self.queue.append(
                    Customer(
                        desired_item=desired, patience_ticks_remaining=patience_ticks
                    )
                )
            else:
                reward -= self.balk_penalty
                self.balked_count += 1

        # Waiting penalties and abandonment
        wait_penalty = self.wait_penalty_per_tick * len(self.queue)
        reward -= wait_penalty
        self.total_wait_customers_ticks += len(self.queue)
        survivors: List[Customer] = []
        for c in self.queue:
            c.patience_ticks_remaining -= 1
            if c.patience_ticks_remaining <= 0:
                reward -= self.abandon_penalty
                self.abandoned_count += 1
            else:
                survivors.append(c)
        self.queue = survivors

        return reward

    # ------ Observation, mask, info ------

    def _obs(self) -> np.ndarray:
        t_frac = self.t / max(1, self.day_ticks - 1)
        time_feats = np.array(
            [math.sin(2 * math.pi * t_frac), math.cos(2 * math.pi * t_frac)],
            dtype=np.float32,
        )

        inv = np.clip(self.inventory, 0, 50).astype(np.float32) / 50.0

        # Oven status as max remaining time fraction in each oven
        oven_fracs = []
        for oven in self.ovens:
            if not oven:
                oven_fracs.append(0.0)
            else:
                max_ticks = max(load[2] for load in oven)
                # Normalize by the longest item bake time in ticks
                denom = int(
                    math.ceil(np.max(ITEM_BAKE_SECONDS) / self.seconds_per_tick)
                )
                oven_fracs.append(min(1.0, max_ticks / max(1, denom)))
        oven_fracs = np.array(oven_fracs, dtype=np.float32)

        q_len = np.array([min(1.0, len(self.queue) / self.queue_cap)], dtype=np.float32)

        # First K customers: item one-hot + patience fraction
        K = self.observe_k_customers
        cust_feats = np.zeros((K, 6), dtype=np.float32)
        for i in range(min(K, len(self.queue))):
            c = self.queue[i]
            cust_feats[i, c.desired_item] = 1.0
            cust_feats[i, 5] = np.clip(c.patience_ticks_remaining / 10.0, 0.0, 1.0)
        cust_flat = cust_feats.flatten()

        price_mul = np.array(
            [np.clip(self.daily_price_multiplier, 0.5, 1.5)], dtype=np.float32
        )

        obs = np.concatenate(
            [time_feats, inv, oven_fracs, q_len, cust_flat, price_mul], dtype=np.float32
        )
        return obs

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)

        # Serve actions: valid if inventory>0 and some customer wants it
        wants = np.zeros(5, dtype=np.int32)
        for c in self.queue:
            wants[c.desired_item] += 1
        for i in range(5):
            mask[i] = (self.inventory[i] > 0) and (wants[i] > 0)

        # Bake actions: valid if any oven has capacity for that item
        for i in range(5):
            size_u = float(ITEM_SIZE[i])
            has_space = False
            for oven in self.ovens:
                used = sum(load[1] for load in oven)
                if used + size_u <= self.oven_capacity:
                    has_space = True
                    break
            mask[5 + i] = has_space

        # If we can serve any customer right now, mask out all bake actions to prioritize service
        if np.any(mask[:5]):
            for i in range(5, 10):
                mask[i] = False

        # Idle always allowed
        mask[-1] = True
        return mask

    def _info(self) -> Dict:
        return {
            "action_mask": self._action_mask(),
            "profit": self.profit,
            "t": self.t,
            "prices_today": self.prices_today.copy(),
            "last_action": self.last_action_str,
            # Evaluation counters
            "served": self.served_count,
            "abandoned": self.abandoned_count,
            "arrivals": self.arrivals_count,
            "balked": self.balked_count,
            "wait_ticks": self.total_wait_customers_ticks,
            "leftover_units": (
                int(self.inventory.sum()) if self.t >= self.day_ticks else None
            ),
            "leftover_cost": (
                float(np.dot(self.inventory, self.costs))
                if self.t >= self.day_ticks
                else None
            ),
        }

    # ------ Utils ------

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        s = float(np.sum(v))
        if s <= 0:
            return np.full_like(v, 1.0 / len(v))
        return v / s



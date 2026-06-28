import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json


# ─────────────────────────────────────────────────────────────────────────────
# CALCE-модель деградации (загружается из degradation_model.json)
# ─────────────────────────────────────────────────────────────────────────────

def _piecewise_linear(n, n_break, a1, slope1, slope2):
    """Кусочно-линейная модель SOH(n) из Этапа 1."""
    return np.where(
        n < n_break,
        a1 - slope1 * n,
        a1 - slope1 * n_break - slope2 * (n - n_break)
    )

def load_degradation_model(json_path: str = "degradation_model.json") -> dict:
    """
    Загружает параметры модели деградации из JSON.
    Если файл не найден — использует fallback-параметры CS2_35.
    """
    try:
        with open(json_path, "r") as f:
            params = json.load(f)
        print(f"[BatteryEnv] Модель деградации загружена из {json_path}")
        return params
    except FileNotFoundError:
        print("[BatteryEnv] WARN: degradation_model.json не найден, используется fallback CS2_35")
        return {
            "CS2_35": {
                "n_break": 613.38,
                "a1":      1.0194,
                "slope1":  9.32e-05,
                "slope2":  2.048e-03,
                "r2":      0.9847
            },
            "CS2_36": {
                "n_break": 656.83,
                "a1":      1.0647,
                "slope1":  2.532e-04,
                "slope2":  2.349e-03,
                "r2":      0.9864
            }
        }

def degradation_factor(params: dict, cell_id: str, global_cycle: float) -> float:
    """
    Возвращает SOH ∈ [0, 1] для батареи cell_id на цикле global_cycle.
    Использует кусочно-линейную модель R²=0.985 из датасета CALCE CS2.
    """
    p = params[cell_id]
    soh = _piecewise_linear(
        global_cycle,
        p["n_break"], p["a1"], p["slope1"], p["slope2"]
    )
    return float(np.clip(soh, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Среда
# ─────────────────────────────────────────────────────────────────────────────

class BatteryEnv(gym.Env):
    """
    Battery Energy Storage System (BESS) среда с физически обоснованной
    моделью деградации на основе данных CALCE CS2.

    Observation space (8 признаков):
        [soc, soh, load, pv, price, hour, global_cycle_norm, degradation_rate]

    Action space:
        [-1, 1] → [-max_charge_power, +max_charge_power] кВт

    Reward:
        -cost - λ · Δsoh_real
        где cost     = стоимость энергии из сети
            Δsoh_real = реальное снижение SOH по модели CALCE
            λ         = degradation_penalty_weight (настраивается)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cell_id: str = "CS2_35",
        degradation_model_path: str = "degradation_model.json",
        degradation_penalty_weight: float = 500.0,
        episode_length: int = 24,
    ):
        super().__init__()

        # ── Параметры батареи ──────────────────────────────────────────────
        self.capacity          = 10.0   # кВт·ч, номинальная ёмкость
        self.max_charge_power  = 2.0    # кВт, максимальная мощность заряда/разряда
        self.efficiency        = 0.9    # КПД round-trip
        self.episode_length    = episode_length

        # ── Модель деградации CALCE ────────────────────────────────────────
        self.cell_id                   = cell_id
        self.degradation_params        = load_degradation_model(degradation_model_path)
        self.degradation_penalty_weight = degradation_penalty_weight

        # ── Пространство действий: [-1, 1] ────────────────────────────────
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # ── Пространство состояний: 8 признаков ───────────────────────────
        # [soc, soh, load, pv, price, hour_norm, global_cycle_norm, deg_rate]
        low  = np.array([0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 5.0, 5.0, 20.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Точка излома из модели — агент "знает" границу ускоренной деградации
        self._n_break = self.degradation_params[cell_id]["n_break"]

        self.reset()

    # ── Reset ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.soc          = 0.5    # начальный заряд 50%
        self.current_step = 0

        # global_cycle: случайный старт из первой половины жизни батареи,
        # чтобы агент видел разные стадии деградации во время обучения
        self.global_cycle = float(
            self.np_random.integers(1, int(self._n_break * 1.5))
            if hasattr(self, "np_random") and self.np_random is not None
            else 1
        )

        # SOH пересчитывается из модели CALCE
        self.soh = degradation_factor(
            self.degradation_params, self.cell_id, self.global_cycle
        )

        return self._get_obs(), {}

    # ── Observation ────────────────────────────────────────────────────────

    def _get_price_and_profile(self, hour: int):
        """Имитация почасового тарифа и профиля нагрузки/генерации."""
        if 18 <= hour <= 22:
            price = 15.0   # пиковый тариф
        elif 0 <= hour <= 6:
            price = 2.0    # ночной тариф
        else:
            price = 7.0    # дневной тариф

        load = 1.0 + np.sin(hour * np.pi / 12) * 0.5
        pv   = 2.0 if 10 <= hour <= 16 else 0.0
        return price, load, pv

    def _get_degradation_rate(self) -> float:
        """
        Нормированная скорость деградации [0, 1].
        До точки излома ≈ 0 (медленная), после ≈ 1 (быстрая).
        Агент использует это как сигнал опасности.
        """
        p = self.degradation_params[self.cell_id]
        if self.global_cycle < self._n_break:
            rate = p["slope1"]
        else:
            rate = p["slope2"]
        # Нормируем: slope2 — максимальная известная скорость
        return float(np.clip(rate / p["slope2"], 0.0, 1.0))

    def _get_obs(self) -> np.ndarray:
        hour = self.current_step % 24
        price, load, pv = self._get_price_and_profile(hour)

        # Нормировки для нейросети
        hour_norm         = hour / 23.0                              # [0, 1]
        global_cycle_norm = np.clip(self.global_cycle / 1000.0, 0.0, 1.0)  # [0, 1]
        degradation_rate  = self._get_degradation_rate()             # [0, 1]

        return np.array([
            self.soc,             # текущий заряд [0, 1]
            self.soh,             # здоровье батареи [0, 1]
            load,                 # потребление кВт
            pv,                   # генерация PV кВт
            price,                # тариф ₽/кВт·ч
            hour_norm,            # час суток нормированный
            global_cycle_norm,    # стадия жизни батареи
            degradation_rate,     # скорость деградации (0=медленно, 1=быстро)
        ], dtype=np.float32)

    # ── Step ───────────────────────────────────────────────────────────────

    def step(self, action):
        hour = self.current_step % 24
        price, load, pv = self._get_price_and_profile(hour)

        # Действие → мощность с учётом КПД
        action_val    = float(np.clip(action[0], -1.0, 1.0))
        applied_power = action_val * self.max_charge_power

        # КПД: при заряде теряем энергию, при разряде — тоже
        if applied_power >= 0:
            effective_power = applied_power * self.efficiency   # заряд: меньше входит
        else:
            effective_power = applied_power / self.efficiency   # разряд: меньше выходит

        # Обновление SOC
        energy_change = effective_power / self.capacity
        self.soc = float(np.clip(self.soc + energy_change, 0.0, 1.0))

        # ── Обновление SOH через модель CALCE ─────────────────────────────
        soh_before = self.soh

        # Каждое действие с батареей = доля цикла
        # Полный разряд/заряд = 1 цикл; частичное использование = меньше
        cycle_fraction = abs(energy_change)  # [0, ~0.2] за один шаг

        self.global_cycle += cycle_fraction
        self.soh = degradation_factor(
            self.degradation_params, self.cell_id, self.global_cycle
        )

        delta_soh = soh_before - self.soh   # реальное снижение SOH [≥ 0]

        # ── Reward ────────────────────────────────────────────────────────
        # Баланс сети: положительный → покупаем из сети (платим)
        grid_flow = load - pv + applied_power
        cost      = max(0.0, grid_flow) * price

        # Штраф за деградацию масштабируется реальным ΔSOH из CALCE
        degradation_penalty = self.degradation_penalty_weight * delta_soh

        reward = -cost - degradation_penalty

        # ── Инфо для отладки ──────────────────────────────────────────────
        info = {
            "cost":               cost,
            "degradation_penalty": degradation_penalty,
            "delta_soh":          delta_soh,
            "soh":                self.soh,
            "soc":                self.soc,
            "global_cycle":       self.global_cycle,
            "grid_flow":          grid_flow,
        }

        self.current_step += 1
        terminated = self.current_step >= self.episode_length

        return self._get_obs(), float(reward), terminated, False, info
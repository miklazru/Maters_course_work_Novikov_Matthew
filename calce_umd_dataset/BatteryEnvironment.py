import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BatteryEnv(gym.Env):

    # граница
    CELL_DATA = {
        "CS2_35": {
            "nominal_capacity": 1.1,
            "a1": 1.0, "slope1": 0.0001, "n_break": 150, "slope2": 0.0005
        },
        "CS2_36": {
            "nominal_capacity": 1.1,
            "a1": 1.02, "slope1": 0.00012, "n_break": 120, "slope2": 0.0006
        }
    }
    # граница
    BATTERY_COST_RUB = 2000.0

    def __init__(self, cell_ids=["CS2_35","CS2_36"]):
        super(BatteryEnv, self).__init__()

        self.cell_ids = cell_ids

        self.batteries = []
        for cid in cell_ids:
            self.batteries.append({'id': cid, 
             'soc': 0.5, 
             'soh': 1.0, 
             'config': self.CELL_DATA[cid],
             'capacity': self.CELL_DATA[cid]["nominal_capacity"]
             })    
    
        self.max_charge_power = 2.0
        self.efficiency = 0.9  

        # Тариф на электричество (₽/кВт·ч) — реальные российские тарифы
        self.TARRIFF = {
            'night': 2.0,   # 00:00–06:00
            'day':   7.0,   # 07:00–17:00
            'peak':  15.0   # 18:00–22:00
        }
        
        # 2. Пространство действий: [-1, 1] 
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 3. Состояние: [SoC, SoH, Load, PV, Price, Hour] - теперь 6 признаков
        self.observation_space = spaces.Box(low=0, high=100, shape=(9,), dtype=np.float32)
        
        self.reset()

    # граница
    def _get_soh_value(self, cycle, config):
        """Piecewise модель из твоего ноутбука"""
        if cycle < config['n_break']:
            return config['a1'] - config['slope1'] * cycle
        else:
            # Значение в точке перелома для непрерывности графика
            val_at_break = config['a1'] - config['slope1'] * config['n_break']
            return val_at_break - config['slope2'] * (cycle - config['n_break'])
        
    def _get_price(self, hour):
        if 18 <= hour <= 22:
            return self.TARRIFF['peak']
        elif 0 <= hour <= 6:
            return self.TARRIFF['night']
        else:
            return self.TARRIFF['day']

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        for bat in self.batteries:
            bat['soc'] = float(np.random.uniform(0.2, 0.8))
            bat['global_cycle'] = 1
            bat['soh'] = self._get_soh_value(1, bat['config'])
        return self._get_obs(), {}

    def _get_obs(self):
        hour = self.current_step % 24
        # Имитация тарифа через отдельную функцию
        price = self._get_price(hour)
        # Имитация баланса
        load = 1.0 + np.sin(hour * np.pi / 12) * 0.5
        pv = 2.0 if 10 <= hour <= 16 else 0.0

        # Часов до начала пика (18ч) — агент "видит" когда выгоднее разрядить
        hours_to_peak = (18 - hour) % 24
        hours_to_peak_norm = hours_to_peak / 24.0  # нормируем [0, 1]

        obs = []
        for bat in self.batteries:
            obs.extend([bat['soc'], bat['soh']])
        
        obs.extend([load, pv, price, float(hour), hours_to_peak_norm])
        # return np.array([self.soc, self.soh, load, pv, price, float(hour), hours_to_peak_norm], dtype=np.float32)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        # 1. Получаем состояние
        obs = self._get_obs()
        # Индексы: [soc1, soh1, soc2, soh2, load, pv, price, hour, h_to_peak]
        load, pv, price = obs[4], obs[5], obs[6] 
        hour = int(obs[7])
        
        total_applied_power = 0
        total_wear_cost = 0
        
        # 2. Обрабатываем обе батареи
        for i, bat in enumerate(self.batteries):
            # Масштабируем действие (-1 до 1) в мощность
            requested_power = float(np.clip(action[i], -1.0, 1.0)) * self.max_charge_power
            
            # Физика заряда/разряда
            if requested_power > 0:
                max_charge = (1.0 - bat['soc']) * bat['capacity']
                applied_power = min(requested_power, max_charge)
            else:
                max_discharge = bat['soc'] * bat['capacity']
                applied_power = -min(abs(requested_power), max_discharge)
            
            # Обновление состояния батареи
            energy_change = applied_power / bat['capacity']
            bat['soc'] = float(np.clip(bat['soc'] + energy_change, 0.0, 1.0))
            bat['global_cycle'] += abs(energy_change)
            # Расчет износа (SoH)
            soh_delta = abs(energy_change) * 0.0003
            bat['soh'] -= soh_delta
            
            # Накопление затрат на износ
            total_wear_cost += soh_delta * self.BATTERY_COST_RUB
            total_applied_power += applied_power

        # 3. Экономическая логика (Reward)
        # Сколько бы мы заплатили без батарей
        cost_without_battery = max(0.0, load - pv) * price
        
        # Сколько мы платим с учетом работы батарей
        cost_with_battery = max(0.0, load - pv + total_applied_power) * price
        
        # Экономия - это то, что мы "выиграли"
        savings = cost_without_battery - cost_with_battery
        
        # Итоговый реворд: Выгода минус штраф за износ
        reward = savings - total_wear_cost
        
        self.current_step += 1

        info = {
            'cost_rub': cost_with_battery,
            'wear_rub': total_wear_cost,
            'savings': savings,
            'reward': reward,
            'soc_35': self.batteries[0]['soc'],
            'soc_36': self.batteries[1]['soc'],
            'soh_35': self.batteries[0]['soh'],
            'soh_36': self.batteries[1]['soh'],
        }
        
        # Возвращаем данные
        return self._get_obs(), float(reward), self.current_step >= 24, False, info
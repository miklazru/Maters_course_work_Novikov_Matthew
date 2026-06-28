import random
import numpy as np

class QLearningAgent:
    def __init__(self, n_soc=11, n_hours=24, n_actions=3):
        # Q-таблица: [SoC][Hour][Action]
        # Действия: 0 (разряд -1), 1 (ничего 0), 2 (заряд 1)
        self.q_table = np.zeros((n_soc, n_hours, n_actions))
        
        self.alpha = 0.1    # Скорость обучения
        self.gamma = 0.95   # Фактор дисконтирования (важность будущего)
        self.epsilon = 1.0  # Вероятность случайного хода (исследование)
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

    def get_action(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randint(0, 2) # Случайное действие
        return np.argmax(self.q_table[state]) # Лучшее из известных

    def learn(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        
        # Формула Беллмана
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value
        
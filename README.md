# ⚡ BESS Management System — Иерархическая архитектура SAC + LLM

> **Курсовой проект** | НИУ ВШЭ, Факультет компьютерных наук, 2026  
> Моделирование экосистемы «Сберсити» на основе распределённых интеллектуальных измерительных устройств

---

## О проекте

Гибридная мультиагентная система управления гетерогенными системами накопления энергии (BESS), объединяющая непрерывное обучение с подкреплением и большие языковые модели для максимизации экономической выгоды и минимизации физического износа литий-ионных батарей.

Система реализует трёхуровневую иерархию управления:

| Слой | Компонент | Роль |
|---|---|---|
| 🔵 **Оперативный** | Soft Actor-Critic (SAC) | Принятие решений заряд/разряд в реальном времени |
| 🟡 **Тактический** | LLM-Координатор (Qwen2.5-3B) | Превентивная защита изношенных ячеек через маскирование действий |
| 🔴 **Стратегический** | LLM-Аналитик (Gemini) | Пост-аудит суточных логов и инженерные рекомендации |

---

## Ключевые особенности

### 🔋 Кастомная среда `BatteryEnv` (Gymnasium)

- Физико-математическая симуляция процессов заряда/разряда
- Эмпирические профили циклического старения (деградации SOH) батарей **CS2\_35** и **CS2\_36** из датасета [CALCE (University of Maryland)](https://calce.umd.edu/battery-data)
- Reward Shaping: конвертация физического износа ($\Delta\text{SOH}$) в финансовые штрафы

### 🤖 Оперативный слой — Soft Actor-Critic

- RL-агент, обученный на исторических данных тарифов и солнечной генерации
- Эффективное сглаживание пиков нагрузки (Load Shaving) и энергетический арбитраж
- **Интегральная прибыль: +41.95 ₽/сутки** — эффективнее Rule-Based эвристик на 44 ₽/сутки

### 🧠 Тактический слой — LLM-Координатор

- Локально развёрнутая модель **Qwen2.5-3B-Instruct** (GGUF, через llama.cpp)
- Динамическое бинарное маскирование пространства действий RL-агента
- Защита изношенных ячеек **без переобучения политики**

### 📊 Стратегический слой — LLM-Аналитик

- Автономный модуль `llm_analyst.py` на базе локально развёрнутой модели **Qwen2.5-3B-Instruct** (GGUF, через llama.cpp)
- Генерация инженерных отчётов в Markdown с рекомендациями по изменению порогов управления

---

## Структура репозитория

```
kursach/
├── CityLearn/                          # Прототип на базе CityLearn + Q-Learning
│   ├── llm_server/
│   │   └── llm_server_bash_script.sh   # Запуск локального Qwen2.5-3B сервера
│   ├── BatteryEnvironment.py           # Ранняя версия среды Gymnasium
│   ├── CityLearn_test.ipynb            # Тестирование CityLearn-интеграции
│   ├── QLearningClass.py               # Реализация Q-Learning агента
│   ├── QLearning_vs_ruleBased.ipynb    # Сравнение Q-Learning и Rule-Based
│   └── requirements.txt
│
├── calce_umd_dataset/                  # Основной модуль проекта (CALCE + SAC + LLM)
│   ├── llm_server/
│   │   └── llm_server_bash_script.sh   # Запуск локального Qwen2.5-3B сервера
│   ├── CS2_33/ CS2_34/ CS2_35/ CS2_36/ # Сырые данные батарей CALCE UMD
│   ├── BatteryEnvironment.py           # Финальная кастомная среда Gymnasium
│   ├── llm_coordinator.py              # Тактический LLM-координатор (SAC + маскирование)
│   ├── llm_analyst.py                  # Стратегический LLM-аналитик (Gemini API)
│   ├── Calce_UMD_3.ipynb              # Финальный ноутбук с экспериментами
│   ├── sac_battery_model.zip           # Обученная модель SAC (single battery)
│   ├── sac_multi_battery_model.zip     # Обученная модель SAC (multi battery)
│   ├── ppo_battery_model.zip           # Обученная модель PPO
│   ├── ocv_lut.csv                     # Look-up table OCV-SOC
│   ├── full_battery_data_final.csv     # Финальный датасет деградации
│   ├── degradation_model.json          # Параметры модели деградации SOH
│   └── requirements.txt
│
├── datasets/                           # Вспомогательные датасеты
├── test_datsets.ipynb                  # Разведочный анализ данных
└── README.md
```

---

## Установка и запуск

### 1. Клонирование репозитория

```bash
git clone https://github.com/your-username/sbercity-bess-rl-llm.git
cd sbercity-bess-rl-llm
```

### 2. Запуск локального LLM-сервера (Тактический координатор)

Скачивает квантованную модель Qwen2.5-3B и поднимает OpenAI-совместимый API на `http://localhost:8000`:

```bash
chmod +x CityLearn/llm_server/llm_server_bash_script.sh
./CityLearn/llm_server/llm_server_bash_script.sh
```

### 3. Обучение и симуляция BESS

```bash
pip install -r requirements.txt
jupyter notebook notebooks/Calce_UMD_3.ipynb
```

### 4. Запуск стратегического LLM-аналитика

```bash
chmod +x calce_umd_dataset/llm_server/llm_server_bash_script.sh
./calce_umd_dataset/llm_server/llm_server_bash_script.sh
```

---

## Результаты

```
Метод                   Прибыль (₽/сутки)
─────────────────────────────────────────
Rule-Based (эвристика)       ~-2.05
SAC (RL-агент)               +41.95   ✅ +44 ₽/сутки vs Rule-Based
Hierarchical SAC + LLM       +41.95   ✅ + защита SOH без потери прибыли
```

---

## Технологический стек

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-blue)
![llama.cpp](https://img.shields.io/badge/llama.cpp-local_LLM-green)
![Qwen](https://img.shields.io/badge/Qwen2.5--3B-Instruct-purple)
![Gemini](https://img.shields.io/badge/Gemini_API-analyst-4285F4?logo=google)

---

## Авторы

**Новиков Матвей Андреевич** — исследование среды BESS, обучение RL-моделей, интеграция LLM-координатора  
**Хасянов Руфат** — Коллега, разрабатывает параллельно свой проект по управлению ситуации внутри батареи

**Научные руководители:** Васильев С.П., Дружинин А.

---

<div align="center">
  <sub>НИУ ВШЭ · Факультет компьютерных наук · 2026</sub>
</div>

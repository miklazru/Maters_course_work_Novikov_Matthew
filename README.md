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

- Автономный модуль `llm_analyst.py` на базе Gemini API
- Генерация инженерных отчётов в Markdown с рекомендациями по изменению порогов управления

---

## Структура репозитория

```
sbercity-bess-rl-llm/
├── envs/
│   └── battery_env.py          # Кастомная среда Gymnasium
├── agents/
│   ├── rule_based.py           # Базовые эвристические стратегии
│   └── sac_train.py            # Обучение Soft Actor-Critic
├── llm_control/
│   ├── start_qwen_server.sh    # Запуск локального Qwen2.5-3B сервера
│   ├── hierarchical_agent.py   # Гибридный контроллер (SAC + LLM Masking)
│   └── llm_analyst.py          # Стратегический аналитик (Gemini API)
├── notebooks/
│   └── Calce_UMD_3.ipynb       # Финальный ноутбук с экспериментами
├── data/
│   └── calce_profiles.json     # Эмпирические данные деградации
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
**Хасянов Руфат** — разработка смежной инфраструктуры

**Научные руководители:** Васильев С.П., Дружинин А.

---

<div align="center">
  <sub>НИУ ВШЭ · Факультет компьютерных наук · 2026</sub>
</div>

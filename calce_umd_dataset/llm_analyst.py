"""
llm_analyst.py
==============

Strategic Analyst Layer для гибридной мультибатарейной системы накопления
энергии (BESS), управляемой алгоритмом RL (SAC) под надзором тактического
LLM-координатора.

Модуль выполняет пост-аудит суточного цикла (24 шага): принимает структурированные
логи, строит контекстный промпт, запрашивает локальную модель Qwen2.5-7B-Instruct
(через OpenAI-совместимый llama-server) и возвращает инженерно-экономический
отчет в формате Markdown.

Использование (программно, например из ноутбука):

    from llm_analyst import generate_strategic_report

    report_md = generate_strategic_report(logs)  # logs — список из 24 dict
    print(report_md)

    # или через класс:
    from llm_analyst import StrategicAnalyst
    analyst = StrategicAnalyst()
    report_md = analyst.generate_report(logs)

Использование из командной строки:

    python llm_analyst.py daily_log.json report.md

Схема входного лога (один dict на каждый час суток, 24 записи):

    hour          int    : час суток [0..23]
    soc_35        float  : SOC батареи CS2_35, [0..1]
    soh_35        float  : SOH батареи CS2_35, [0..1]
    soc_36        float  : SOC батареи CS2_36, [0..1]
    soh_36        float  : SOH батареи CS2_36, [0..1]
    load          float  : базовая нагрузка сети, кВт
    pv            float  : солнечная генерация, кВт
    price         float  : тариф сети, руб/кВт·ч
    action_35     float  : команда SAC для CS2_35, [-1..1]
    action_36     float  : команда SAC для CS2_36, [-1..1]
    cost_with     float  : затраты на сеть с АКБ, руб
    cost_without  float  : затраты на сеть без АКБ, руб
    wear_cost     float  : оценка износа за шаг, руб
    reward        float  : итоговая награда за шаг, руб
"""

import json
import os
import time

from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Конфигурация — локальный llama-server, Qwen2.5-7B-Instruct (порт 8001)
# ─────────────────────────────────────────────────────────────────────────────

QWEN_ANALYST_BASE_URL = "http://127.0.0.1:8001/v1"
QWEN_ANALYST_MODEL = "Qwen2.5-3B-Instruct-Q5_K_M.gguf" 
# QWEN_ANALYST_MODEL = "qwen2.5-1.5b-instruct-q8_0.gguf" 

# Задержки экспоненциального отката (секунды) между попытками запроса
BACKOFF_DELAYS = [1, 2, 4, 8, 16]
MAX_RETRIES_DEFAULT = 1

REQUIRED_FIELDS = [
    "hour", "soc_35", "soh_35", "soc_36", "soh_36",
    "load", "pv", "price", "action_35", "action_36",
    "cost_with", "cost_without", "wear_cost", "reward",
]

PEAK_HOURS = range(18, 23)   # 18:00–22:00, тариф 15 руб/кВт·ч
SOC_CRITICAL_THRESHOLD = 0.20
SOC_EMPTY_THRESHOLD = 0.05
IDLE_ACTION_THRESHOLD = 0.02

SYSTEM_INSTRUCTION = (
    "Ты — ведущий системный аналитик промышленных накопителей энергии (BESS) "
    "и систем Smart Grid. Твоя задача — проводить строгий аудит суточных "
    "циклов работы гетерогенных батарей на основе предоставленных технических "
    "и финансовых логов. Твой тон — профессиональный, инженерный, критический. "
    "Ты должен оперировать точными цифрами, выявлять неэффективность алгоритмов "
    "управления, находить скрытые аномалии деградации ячеек и давать конкретные "
    "математические рекомендации по изменению порогов тактического управления."
)

REPORT_INSTRUCTIONS = """
Сформируй КРАТКИЙ отчет в Markdown. ВАЖНО:
- НЕ переписывай и не пересказывай цифры из сводки выше — она уже дана пользователю.
- Только анализ, причины и выводы. Каждый раздел — максимум 2-3 коротких предложения.
- Без подзаголовков на каждую метрику, без жирного шрифта на каждое число.

## Раздел 1. Баланс суток
Одной фразой: насколько эффективна система (экономия минус износ) и хорошо ли
срезаны пики.

## Раздел 2. SOC/SOH аномалии
Если есть микроциклирование, критический разряд или пустая батарея в пик —
назови причину и последствие. Если аномалий нет — напиши это одной фразой.

## Раздел 3. Критика управления
Если батарея пустая в пиковые часы (см. сводку) — это означает, что система
"выстрелила" заряд слишком рано и не дотянула до пика. Объясни почему это
плохо и что теряется.

## Раздел 4. Рекомендации
Дай ровно 2 конкретные рекомендации (порог SOH для маскирования или
коэффициент штрафа в reward), без воды.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Валидация входных данных
# ─────────────────────────────────────────────────────────────────────────────

def _validate_logs(logs):
    if not logs:
        raise ValueError(
            "Лог пуст: ожидается список словарей (обычно 24 записи, "
            "по одной на каждый час суток)."
        )

    missing_overall = set()
    for row in logs:
        missing_overall.update(f for f in REQUIRED_FIELDS if f not in row)

    if missing_overall:
        raise ValueError(
            f"В логе отсутствуют обязательные поля: {sorted(missing_overall)}.\n"
            f"Ожидаемая схема: {REQUIRED_FIELDS}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Расчет сводных показателей (программно, чтобы LLM не придумывала суммы)
# ─────────────────────────────────────────────────────────────────────────────

def _count_sign_flips(rows, key):
    flips = 0
    prev_sign = None
    for r in rows:
        val = r[key]
        if abs(val) < IDLE_ACTION_THRESHOLD:
            continue
        sign = 1 if val > 0 else -1
        if prev_sign is not None and sign != prev_sign:
            flips += 1
        prev_sign = sign
    return flips


def _compute_summary(logs):
    rows = sorted(logs, key=lambda r: r["hour"])

    total_cost_with = sum(r["cost_with"] for r in rows)
    total_cost_without = sum(r["cost_without"] for r in rows)
    total_wear = sum(r["wear_cost"] for r in rows)
    total_reward = sum(r["reward"] for r in rows)
    net_savings = total_cost_without - total_cost_with
    net_benefit = net_savings - total_wear

    peak_rows = [r for r in rows if r["hour"] in PEAK_HOURS]
    peak_cost_with = sum(r["cost_with"] for r in peak_rows)
    peak_cost_without = sum(r["cost_without"] for r in peak_rows)

    return {
        "total_cost_with": total_cost_with,
        "total_cost_without": total_cost_without,
        "net_savings": net_savings,
        "total_wear": total_wear,
        "net_benefit": net_benefit,
        "total_reward": total_reward,
        "peak_cost_with": peak_cost_with,
        "peak_cost_without": peak_cost_without,
        "soc_35_min": min(r["soc_35"] for r in rows),
        "soc_36_min": min(r["soc_36"] for r in rows),
        "soh_35_final": rows[-1]["soh_35"],
        "soh_36_final": rows[-1]["soh_36"],
        "flips_35": _count_sign_flips(rows, "action_35"),
        "flips_36": _count_sign_flips(rows, "action_36"),
        "empty_during_peak_35": any(r["soc_35"] < SOC_EMPTY_THRESHOLD for r in peak_rows),
        "empty_during_peak_36": any(r["soc_36"] < SOC_EMPTY_THRESHOLD for r in peak_rows),
        "critical_discharge_35": any(r["soc_35"] < SOC_CRITICAL_THRESHOLD for r in rows),
        "critical_discharge_36": any(r["soc_36"] < SOC_CRITICAL_THRESHOLD for r in rows),
        "idle_hours_35": sum(1 for r in rows if abs(r["action_35"]) < IDLE_ACTION_THRESHOLD),
        "idle_hours_36": sum(1 for r in rows if abs(r["action_36"]) < IDLE_ACTION_THRESHOLD),
    }


def _format_log_table(logs):
    rows = sorted(logs, key=lambda r: r["hour"])
    header = (
        f"{'Час':>3} | {'SOC1':>5} {'SOH1':>6} | {'SOC2':>5} {'SOH2':>6} | "
        f"{'Load':>5} {'PV':>5} {'Цена':>5} | {'Act1':>6} {'Act2':>6} | "
        f"{'Cost-АКБ':>9} {'Cost+АКБ':>9} {'Износ':>7} {'Reward':>7}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{int(r['hour']):>3d} | "
            f"{r['soc_35']:>5.2f} {r['soh_35']:>6.4f} | "
            f"{r['soc_36']:>5.2f} {r['soh_36']:>6.4f} | "
            f"{r['load']:>5.2f} {r['pv']:>5.2f} {r['price']:>5.1f} | "
            f"{r['action_35']:>6.2f} {r['action_36']:>6.2f} | "
            f"{r['cost_without']:>9.2f} {r['cost_with']:>9.2f} "
            f"{r['wear_cost']:>7.3f} {r['reward']:>7.2f}"
        )
    return "\n".join(lines)


def _build_user_prompt(logs):
    summary = _compute_summary(logs)

    summary_text = (
        "СВОДНЫЕ ПОКАЗАТЕЛИ СУТОК (точные значения, рассчитаны программно):\n"
        f"- Затраты без АКБ / с АКБ: {summary['total_cost_without']:.2f} / "
        f"{summary['total_cost_with']:.2f} руб\n"
        f"- Чистая экономия: {summary['net_savings']:.2f} руб, "
        f"износ: {summary['total_wear']:.2f} руб, "
        f"итоговая выгода: {summary['net_benefit']:.2f} руб\n"
        f"- Затраты в пик (18-22ч) без АКБ / с АКБ: "
        f"{summary['peak_cost_without']:.2f} / {summary['peak_cost_with']:.2f} руб\n"
        f"- Минимальный SOC: CS2_35={summary['soc_35_min']:.2f}, "
        f"CS2_36={summary['soc_36_min']:.2f}\n"
        f"- Финальный SOH: CS2_35={summary['soh_35_final']:.3f}, "
        f"CS2_36={summary['soh_36_final']:.3f}\n"
        f"- Смен знака действия (микроциклирование): CS2_35={summary['flips_35']}, "
        f"CS2_36={summary['flips_36']}\n"
        f"- Батарея пуста в пиковые часы: CS2_35={summary['empty_during_peak_35']}, "
        f"CS2_36={summary['empty_during_peak_36']}\n"
        f"- Критический разряд (SOC<0.2): CS2_35={summary['critical_discharge_35']}, "
        f"CS2_36={summary['critical_discharge_36']}\n"
        f"- Часов простоя: CS2_35={summary['idle_hours_35']}, "
        f"CS2_36={summary['idle_hours_36']}"
    )

    return (
        "Суточный лог (24ч) гибридной BESS с ячейками CS2_35 и CS2_36, "
        "управляемой SAC под надзором LLM-координатора.\n\n"
        f"{summary_text}\n\n"
        f"{REPORT_INSTRUCTIONS}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Запрос к локальному Qwen (OpenAI-совместимый llama-server, порт 8001)
# ─────────────────────────────────────────────────────────────────────────────

def _call_qwen_api(prompt, base_url=QWEN_ANALYST_BASE_URL, model=QWEN_ANALYST_MODEL,
                    max_retries=MAX_RETRIES_DEFAULT, timeout=600,
                    max_tokens=600, temperature=0.3):
    client = OpenAI(base_url=base_url, api_key="not-needed")

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            text = response.choices[0].message.content
            if not text or not text.strip():
                raise RuntimeError("Модель вернула пустой ответ.")
            return text

        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                wait = BACKOFF_DELAYS[min(attempt, len(BACKOFF_DELAYS) - 1)]
                print(
                    f"[llm_analyst] Ошибка запроса к {base_url} ({exc}), "
                    f"повтор через {wait}с (попытка {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait)
                continue

    raise RuntimeError(
        f"Не удалось получить ответ от локального Qwen-сервера ({base_url}, "
        f"модель {model}) после {max_retries} попыток: {last_error}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Главная точка входа
# ─────────────────────────────────────────────────────────────────────────────

def generate_strategic_report(logs, base_url=QWEN_ANALYST_BASE_URL,
                                model=QWEN_ANALYST_MODEL,
                                max_retries=MAX_RETRIES_DEFAULT, verbose=True):
    """
    Выполняет полный пост-аудит суточного цикла BESS и возвращает отчет
    в формате Markdown.

    Параметры
    ---------
    logs : list[dict]
        Список записей по часам суток (см. схему в начале файла).
    base_url : str
        Адрес OpenAI-совместимого llama-server с моделью-аналитиком
        (по умолчанию локальный Qwen2.5-7B на порту 8001).
    model : str
        Идентификатор модели на сервере (как в /v1/models).
    max_retries : int
        Максимум попыток запроса при сетевых ошибках.
    verbose : bool
        Печатать ли статус выполнения в консоль.

    Возвращает
    ----------
    str — отчет в формате Markdown.
    """
    if verbose:
        print(f"[llm_analyst] Валидация входного лога ({len(logs)} записей)...")
    _validate_logs(logs)

    if verbose:
        print("[llm_analyst] Формирование контекстного промпта...")
    prompt = _build_user_prompt(logs)

    if verbose:
        print(f"[llm_analyst] Запрос к {model} ({base_url})...")
    report = _call_qwen_api(prompt, base_url=base_url, model=model, max_retries=max_retries)

    if verbose:
        print("[llm_analyst] Отчет получен.")
    return report


def _ask_section(question, base_url, model, max_retries=MAX_RETRIES_DEFAULT,
                  max_tokens=150, temperature=0.2, timeout=300):
    """Узкий запрос на один короткий ответ — модель меньше путается и реже игнорирует формат."""
    strict_suffix = (
        "\n\nОтветь МАКСИМУМ двумя предложениями, обычным текстом. "
        "БЕЗ markdown-таблиц, БЕЗ списков, БЕЗ жирного шрифта, БЕЗ заголовков."
    )
    text = _call_qwen_api(
        question + strict_suffix,
        base_url=base_url, model=model,
        max_retries=max_retries, timeout=timeout,
        max_tokens=max_tokens, temperature=temperature,
    )
    return text.strip()


def generate_strategic_report_decomposed(logs, base_url=QWEN_ANALYST_BASE_URL,
                                           model=QWEN_ANALYST_MODEL,
                                           max_retries=MAX_RETRIES_DEFAULT, verbose=True):
    """
    Альтернатива generate_strategic_report для слабых моделей (например,
    локальный Qwen2.5-1.5B): вместо одного большого запроса на отчет из 4
    разделов делает 4 отдельных узких запроса, каждый с одним простым
    вопросом. Заголовки разделов фиксированы в коде — структура отчета
    гарантирована независимо от того, насколько хорошо модель следует
    форматированию.
    """
    if verbose:
        print(f"[llm_analyst] Валидация входного лога ({len(logs)} записей)...")
    _validate_logs(logs)
    s = _compute_summary(logs)

    q1 = (
        f"Затраты на сеть без АКБ за сутки: {s['total_cost_without']:.2f} руб, "
        f"с АКБ: {s['total_cost_with']:.2f} руб. Чистая экономия: {s['net_savings']:.2f} руб. "
        f"Износ батарей: {s['total_wear']:.2f} руб. Итоговая выгода (экономия минус износ): "
        f"{s['net_benefit']:.2f} руб. В пиковые часы (18-22ч) затраты на сеть снизились с "
        f"{s['peak_cost_without']:.2f} до {s['peak_cost_with']:.2f} руб.\n\n"
        "Оцени: насколько экономически эффективна система и хорошо ли срезаны "
        "пиковые затраты?"
    )

    q2 = (
        f"Финальный SOH: CS2_35={s['soh_35_final']:.3f}, CS2_36={s['soh_36_final']:.3f}. "
        f"Число смен знака действия за сутки (микроциклирование): "
        f"CS2_35={s['flips_35']}, CS2_36={s['flips_36']}. "
        f"Минимальный SOC за сутки: CS2_35={s['soc_35_min']:.2f}, CS2_36={s['soc_36_min']:.2f}. "
        f"Критический разряд (SOC<0.2): CS2_35={s['critical_discharge_35']}, "
        f"CS2_36={s['critical_discharge_36']}. "
        f"Батарея пустая в пиковые часы: CS2_35={s['empty_during_peak_35']}, "
        f"CS2_36={s['empty_during_peak_36']}.\n\n"
        "Оцени: есть ли асимметрия износа между батареями, и есть ли проблемные "
        "циклы (микроциклирование, критический разряд, пустая батарея в пик)?"
    )

    q3 = (
        f"Часов простоя из 24 (батарея не использовалась тактическим LLM-диспетчером): "
        f"CS2_35={s['idle_hours_35']}ч, CS2_36={s['idle_hours_36']}ч.\n\n"
        "Оцени: не 'парализовал' ли тактический слой одну из батарей, отключив её "
        "слишком надолго в ущерб экономии?"
    )

    q4 = (
        f"Итоговая выгода за сутки: {s['net_benefit']:.2f} руб, износ: {s['total_wear']:.2f} руб. "
        f"Простой: CS2_35={s['idle_hours_35']}ч, CS2_36={s['idle_hours_36']}ч из 24. "
        f"Критический разряд: CS2_35={s['critical_discharge_35']}, "
        f"CS2_36={s['critical_discharge_36']}.\n\n"
        "Дай РОВНО одну конкретную рекомендацию по изменению порога SOH для "
        "маскирования батарей или коэффициента штрафа за износ в функции награды."
    )

    sections = [
        ("Раздел 1. Технико-экономический баланс суток", q1),
        ("Раздел 2. Анализ динамики распределения ресурсов (SOC & SOH)", q2),
        ("Раздел 3. Аудит тактического LLM-диспетчера", q3),
        ("Раздел 4. Предписания по оптимизации", q4),
    ]

    parts = []
    for i, (title, question) in enumerate(sections, start=1):
        if verbose:
            print(f"[llm_analyst] Запрос {i}/4: {title}...")
        answer = _ask_section(question, base_url, model, max_retries=max_retries)
        parts.append(f"## {title}\n{answer}")

    if verbose:
        print("[llm_analyst] Отчет собран.")
    return "\n\n".join(parts)


class StrategicAnalyst:
    """
    Объектно-ориентированная обёртка над generate_strategic_report — для
    единообразия с тактическим координатором (llm_coordinator.py).

    Параметр decomposed=True (по умолчанию) использует 4 отдельных узких
    запроса вместо одного большого — надежнее для слабых локальных моделей
    (например, Qwen2.5-1.5B), которые плохо держат сложное форматирование
    в одном ответе.

    Использование:

        analyst = StrategicAnalyst()
        report_md = analyst.generate_report(daily_logs)
    """

    def __init__(self, base_url=QWEN_ANALYST_BASE_URL, model=QWEN_ANALYST_MODEL,
                 max_retries=MAX_RETRIES_DEFAULT, verbose=True, decomposed=True):
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.verbose = verbose
        self.decomposed = decomposed

    def generate_report(self, logs):
        """Принимает список почасовых записей за сутки, возвращает Markdown-отчет."""
        if self.decomposed:
            return generate_strategic_report_decomposed(
                logs,
                base_url=self.base_url,
                model=self.model,
                max_retries=self.max_retries,
                verbose=self.verbose,
            )
        return generate_strategic_report(
            logs,
            base_url=self.base_url,
            model=self.model,
            max_retries=self.max_retries,
            verbose=self.verbose,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI: python llm_analyst.py <log.json> [report.md]
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Использование: python llm_analyst.py <путь_к_логу.json> [путь_к_отчету.md]")
        sys.exit(1)

    log_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "strategic_report.md"

    with open(log_path, "r", encoding="utf-8") as f:
        logs_data = json.load(f)

    report_text = generate_strategic_report(logs_data)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nОтчет сохранен: {out_path}")

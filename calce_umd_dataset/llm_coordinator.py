"""
llm_coordinator.py
===================

Тактический LLM-координатор (Tactical Coordinator Layer).

Оборачивает базового SAC-агента бинарным маскированием: на каждом шаге
LLM (Qwen, через локальный llama-server) решает, какой из батарей разрешено
исполнять действие SAC в этот час, а какой — нет (mask = 0).

Это ИНСТАНЦИЯ "быстрого" тактического слоя — в отличие от llm_analyst.py,
который выполняет медленный пост-аудит суток целиком (Gemini), этот класс
работает онлайн, по одному часу за раз, поверх уже обученной политики SAC.

Использование:

    from llm_coordinator import HierarchicalLLMSACController

    def ask_qwen(prompt, system_role="..."):
        ...  # твоя текущая функция запроса к llama-server

    hybrid_agent = HierarchicalLLMSACController(
        sac_model=modelSAC,
        ask_fn=ask_qwen,
    )

    history_hybrid = run_episode_detailed(env, hybrid_agent.select_action)
"""

import json
import re

import numpy as np

DEFAULT_SYSTEM_ROLE = (
    "You are a machine that outputs only raw JSON. Never write text explanations."
)

DEFAULT_PROMPT_TEMPLATE = """\
B1_SOC: {soc1:.2f}, B1_SOH: {soh1:.2f}
B2_SOC: {soc2:.2f}, B2_SOH: {soh2:.2f}
Load: {load:.2f}, Price: {price:.2f}, Hour: {hour:.0f}

Decide which battery should be ACTIVE this hour based on the data above
(higher SOC and higher SOH should generally be preferred; price and hour
indicate whether this is a cheap/charging period or expensive/peak period).

Output ONLY a JSON object with this exact schema, replacing X and Y with
your decision (each must be either the integer 0 or the integer 1, and at
least one of them must be 1):
{{"mask_b1": X, "mask_b2": Y}}

Do not write any prose, markdown block, or explanation. Just the JSON object
with YOUR computed values for X and Y based on THIS hour's data above."""


class HierarchicalLLMSACController:
    """
    Иерархический контроллер: SAC выдаёт базовое непрерывное действие,
    LLM-координатор поверх него накладывает бинарную маску [0, 1] на каждую
    батарею для данного часа.

    Параметры
    ---------
    sac_model : stable_baselines3 model
        Обученная модель SAC (или совместимая по интерфейсу .predict()).
    ask_fn : Callable[[str, str], str]
        Функция запроса к LLM: ask_fn(prompt, system_role) -> str.
        Внедряется явно, чтобы класс не был привязан к конкретному клиенту
        (Qwen / llama-server / другой OpenAI-совместимый сервер).
    prompt_template : str
        Шаблон промпта с плейсхолдерами soc1, soh1, soc2, soh2, load, price, hour.
    system_role : str
        Системная роль для LLM.
    verbose : bool
        Печатать ли сырые ответы LLM и статус парсинга в консоль.
    """

    def __init__(
        self,
        sac_model,
        ask_fn,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        system_role: str = DEFAULT_SYSTEM_ROLE,
        verbose: bool = True,
    ):
        self.sac = sac_model
        self.ask_fn = ask_fn
        self.prompt_template = prompt_template
        self.system_role = system_role
        self.verbose = verbose

    def _build_prompt(self, obs) -> str:
        # obs = [soc1, soh1, soc2, soh2, load, pv, price, hour, h_to_peak]
        return self.prompt_template.format(
            soc1=obs[0], soh1=obs[1],
            soc2=obs[2], soh2=obs[3],
            load=obs[4], price=obs[6], hour=obs[7],
        )

    @staticmethod
    def _parse_mask(response: str):
        """Возвращает (mask_b1, mask_b2) или вызывает исключение при сбое парсинга."""
        clean = response.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*?\}", clean, re.DOTALL)
        data = json.loads(match.group(0)) if match else json.loads(clean)
        return float(data["mask_b1"]), float(data["mask_b2"])

    def _fallback_mask(self, obs):
        """Аварийное чередование батарей, если LLM не вернула валидный JSON."""
        hour = int(obs[7])
        if hour % 2 == 0:
            return np.array([1.0, 0.0])
        return np.array([0.0, 1.0])

    def select_action(self, obs):
        sac_action, _ = self.sac.predict(obs, deterministic=True)

        prompt = self._build_prompt(obs)
        response = self.ask_fn(prompt, self.system_role)

        if self.verbose:
            print(f"Час {obs[7]:.0f} | Сырой ответ LLM: {response.strip()}")

        try:
            mask_b1, mask_b2 = self._parse_mask(response)
            if self.verbose:
                print(f"   -> Маска применена: [{mask_b1}, {mask_b2}]")
            return sac_action * np.array([mask_b1, mask_b2])
        except Exception as exc:
            if self.verbose:
                print(f"   !!! Сбой парсинга: {exc}. Аварийное чередование.")
            return sac_action * self._fallback_mask(obs)

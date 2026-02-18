"""
Prompting Templates for Multilingual Chain-of-Thought

Implements controlled language conditions defined by (P_L, T_L, A_L):
- P_L: Problem language
- T_L: Thinking (reasoning) language
- A_L: Answer language

Four primary conditions:
1. Native: (L, L, L) - Problem, thinking, answer all in language L
2. English Baseline: (en, en, en) - Everything in English
3. English Thinking: (L, en, L) - Problem in L, think in English, answer in L
4. Translate-Solve: (L→en, en, en) - Translate problem to English, solve in English
"""

from dataclasses import dataclass
from typing import Optional


# Human-readable language names (duplicated here to avoid circular imports)
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "th": "Thai",
    "sw": "Swahili",
    "bn": "Bengali",
    "te": "Telugu",
}


@dataclass
class LanguageCondition:
    """
    Defines a language condition for the experiment.

    Attributes:
        name: Short identifier (e.g., "native", "en_thinking")
        problem_lang: Language of the problem (or "translate" for translate-solve)
        thinking_lang: Language for chain-of-thought reasoning
        answer_lang: Language for the final answer
        description: Human-readable description
    """

    name: str
    problem_lang: str  # The language the problem is presented in
    thinking_lang: str  # The language to reason in
    answer_lang: str  # The language for the final answer
    description: str

    def __str__(self) -> str:
        return f"{self.name}: ({self.problem_lang}, {self.thinking_lang}, {self.answer_lang})"


# Predefined conditions
CONDITION_NATIVE = LanguageCondition(
    name="native",
    problem_lang="same",  # Same as problem's original language
    thinking_lang="same",
    answer_lang="same",
    description="Native reasoning: problem, thinking, and answer all in the same language",
)

CONDITION_ENGLISH_BASELINE = LanguageCondition(
    name="english_baseline",
    problem_lang="en",
    thinking_lang="en",
    answer_lang="en",
    description="English baseline: everything in English",
)

CONDITION_ENGLISH_THINKING = LanguageCondition(
    name="english_thinking",
    problem_lang="same",  # Original problem language
    thinking_lang="en",  # Reason in English
    answer_lang="same",  # Answer in problem language
    description="English thinking: problem in native language, reason in English, answer in native language",
)

CONDITION_TRANSLATE_SOLVE = LanguageCondition(
    name="translate_solve",
    problem_lang="translate_to_en",
    thinking_lang="en",
    answer_lang="en",
    description="Translate-solve: translate problem to English, then solve entirely in English",
)


# Language-specific instruction fragments
THINKING_INSTRUCTIONS = {
    "en": "Think step by step in English.",
    "es": "Piensa paso a paso en español.",
    "fr": "Réfléchis étape par étape en français.",
    "de": "Denke Schritt für Schritt auf Deutsch.",
    "ru": "Думай шаг за шагом на русском языке.",
    "zh": "请用中文一步一步思考。",
    "ja": "日本語でステップバイステップで考えてください。",
    "th": "คิดทีละขั้นตอนเป็นภาษาไทย",
    "sw": "Fikiria hatua kwa hatua kwa Kiswahili.",
    "bn": "বাংলায় ধাপে ধাপে চিন্তা করুন।",
    "te": "తెలుగులో అడుగు అడుగునా ఆలోచించండి.",
}

ANSWER_INSTRUCTIONS = {
    "en": "Give your final answer after 'Final:' as just the number.",
    "es": "Da tu respuesta final después de 'Final:' como solo el número.",
    "fr": "Donne ta réponse finale après 'Final:' comme juste le nombre.",
    "de": "Gib deine endgültige Antwort nach 'Final:' als nur die Zahl an.",
    "ru": "Дай свой окончательный ответ после 'Final:' как только число.",
    "zh": "在'Final:'后给出你的最终答案，只写数字。",
    "ja": "'Final:'の後に最終的な答えを数字だけで書いてください。",
    "th": "ให้คำตอบสุดท้ายของคุณหลัง 'Final:' เป็นตัวเลขเท่านั้น",
    "sw": "Toa jibu lako la mwisho baada ya 'Final:' kama nambari tu.",
    "bn": "'Final:' এর পরে শুধু সংখ্যা হিসেবে আপনার চূড়ান্ত উত্তর দিন।",
    "te": "'Final:' తర్వాత మీ చివరి సమాధానం సంఖ్యగా మాత్రమే ఇవ్వండి.",
}

TRANSLATE_INSTRUCTION = (
    "First, translate the problem to English. "
    "Then solve it step by step in English. "
    "Give your final answer after 'Final:' as just the number."
)


def build_cot_prompt(
    question: str,
    problem_lang: str,
    condition: LanguageCondition,
    few_shot_examples: Optional[list[dict]] = None,
) -> str:
    """
    Build a chain-of-thought prompt with explicit language control.

    Args:
        question: The math problem text
        problem_lang: The language of the problem (e.g., "es", "fr")
        condition: The LanguageCondition defining (P_L, T_L, A_L)
        few_shot_examples: Optional list of {"question": str, "answer": str} dicts

    Returns:
        Formatted prompt string with language instructions
    """
    parts = []

    # Determine actual languages based on condition
    if condition.thinking_lang == "same":
        thinking_lang = problem_lang
    else:
        thinking_lang = condition.thinking_lang

    if condition.answer_lang == "same":
        answer_lang = problem_lang
    else:
        answer_lang = condition.answer_lang

    # System instruction based on condition
    if condition.name == "translate_solve":
        parts.append(TRANSLATE_INSTRUCTION)
    else:
        # Thinking instruction
        thinking_instruction = THINKING_INSTRUCTIONS.get(
            thinking_lang,
            f"Think step by step in {LANGUAGE_NAMES.get(thinking_lang, thinking_lang)}.",
        )
        parts.append(thinking_instruction)

        # Answer instruction (always use the target answer language)
        answer_instruction = ANSWER_INSTRUCTIONS.get(
            answer_lang,
            "Give your final answer after 'Final:' as just the number.",
        )
        parts.append(answer_instruction)

    # Add few-shot examples if provided
    if few_shot_examples:
        parts.append("")
        for ex in few_shot_examples:
            parts.append(f"Problem: {ex['question']}")
            parts.append(f"{ex['reasoning']}")
            parts.append(f"Final: {ex['answer']}")
            parts.append("")

    # Add the actual problem
    parts.append(f"Problem: {question}")

    return "\n".join(parts)


def build_system_prompt(
    condition: LanguageCondition,
    problem_lang: str,
) -> str:
    """
    Build a system prompt for chat-based models.

    Args:
        condition: The LanguageCondition
        problem_lang: The language of the problem

    Returns:
        System prompt string
    """
    if condition.thinking_lang == "same":
        thinking_lang = problem_lang
    else:
        thinking_lang = condition.thinking_lang

    if condition.answer_lang == "same":
        answer_lang = problem_lang
    else:
        answer_lang = condition.answer_lang

    if condition.name == "translate_solve":
        return (
            "You are a helpful math tutor. When given a problem in a non-English language, "
            "first translate it to English, then solve it step by step in English. "
            "Always end with 'Final:' followed by just the numeric answer."
        )

    thinking_name = LANGUAGE_NAMES.get(thinking_lang, thinking_lang)
    answer_name = LANGUAGE_NAMES.get(answer_lang, answer_lang)

    return (
        f"You are a helpful math tutor. "
        f"Solve problems step by step, reasoning in {thinking_name}. "
        f"Always end with 'Final:' followed by just the numeric answer"
        + (f" in {answer_name}." if answer_lang != thinking_lang else ".")
    )


# Few-shot examples for different languages (basic examples)
FEW_SHOT_EXAMPLES = {
    "en": [
        {
            "question": "Sarah has 5 apples. She buys 3 more. How many apples does she have?",
            "reasoning": "Sarah starts with 5 apples. She buys 3 more apples. Total apples = 5 + 3 = 8.",
            "answer": "8",
        }
    ],
    "es": [
        {
            "question": "Sara tiene 5 manzanas. Compra 3 más. ¿Cuántas manzanas tiene?",
            "reasoning": "Sara empieza con 5 manzanas. Compra 3 manzanas más. Total de manzanas = 5 + 3 = 8.",
            "answer": "8",
        }
    ],
    "fr": [
        {
            "question": "Sarah a 5 pommes. Elle en achète 3 de plus. Combien de pommes a-t-elle?",
            "reasoning": "Sarah commence avec 5 pommes. Elle achète 3 pommes de plus. Total de pommes = 5 + 3 = 8.",
            "answer": "8",
        }
    ],
}

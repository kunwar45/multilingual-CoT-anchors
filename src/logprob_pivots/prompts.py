# ABOUTME: Prompt builders for the two core conditions: prompt_target_cot (reason in the problem's language) and prompt_en_cot (reason in English).
# ABOUTME: Both require the final numeric answer as `FINAL: <number>` so answers parse uniformly across languages.
def prompt_target_cot(question: str, lang: str) -> str:
    """
    “Think in the same language as the question” condition.

    The model is asked to reason step by step in `lang` and output the final
    numeric answer as `FINAL: <number>`.
    """
    return f"""You are a careful mathematician.
Solve the problem step by step in {lang}, then give the final answer as: FINAL: <number>

Problem:
{question}
"""


def prompt_en_cot(question: str) -> str:
    """
    “Think in English even if the question is not English” condition.
    """
    return f"""You are a careful mathematician.
Solve the problem step by step in English, then give the final answer as: FINAL: <number>

Problem:
{question}
"""



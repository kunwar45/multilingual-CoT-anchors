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



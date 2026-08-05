"""
Multilingual Prompts for CoT Experiments

Includes:
- Base solution prompts per language (MGSM and MMATH)
- Rollout prompts (continuation after chunk removal)
- DAG labeling prompt (copied from root prompts.py)
"""

from multicot.data_loaders import Problem, LANGUAGE_NAMES

# ---------------------------------------------------------------------------
# DAG Prompt (verbatim copy from root prompts.py)
# ---------------------------------------------------------------------------

DAG_PROMPT = """
You are an expert in interpreting how language models solve math problems using multi-step reasoning. Your task is to analyze a Chain-of-Thought (CoT) reasoning trace, broken into discrete text chunks, and label each chunk with:

1. **function_tags**: One or more labels that describe what this chunk is *doing* functionally in the reasoning process.

2. **depends_on**: A list of earlier chunk indices that this chunk directly depends on — meaning it uses information, results, or logic introduced in those earlier chunks.

This annotation will be used to build a dependency graph and perform causal analysis, so please be precise and conservative: only mark a chunk as dependent on another if its reasoning clearly uses a previous step's result or idea.

---

### Function Tags (you may assign multiple per chunk if appropriate):

1. `problem_setup`:
    Parsing or rephrasing the problem (initial reading or comprehension).

2. `plan_generation`:
    Stating or deciding on a plan of action (often meta-reasoning).

3. `fact_retrieval`:
    Recalling facts, formulas, problem details (without immediate computation).

4. `active_computation`:
    Performing algebra, calculations, manipulations toward the answer.

5. `result_consolidation`:
    Aggregating intermediate results, summarizing, or preparing final answer.

6. `uncertainty_management`:
    Expressing confusion, re-evaluating, proposing alternative plans (includes backtracking).

7. `final_answer_emission`:
    Explicit statement of the final boxed answer or earlier chunks that contain the final answer.

8. `self_checking`:
    Verifying previous steps, Pythagorean checking, re-confirmations.

9. `unknown`:
    Use only if the chunk does not fit any of the above tags or is purely stylistic or semantic.

---

### depends_on Instructions:

For each chunk, include a list of earlier chunk indices that the reasoning in this chunk *uses*. For example:
- If Chunk 9 performs a computation based on a plan in Chunk 4 and a recalled rule in Chunk 5, then `depends_on: [4, 5]`
- If Chunk 24 plugs in a final answer to verify correctness from Chunk 23, then `depends_on: [23]`
- If there's no clear dependency (e.g. a general plan or recall), use an empty list: `[]`
- If Chunk 13 performs a computation based on information in Chunk 11, which in turn uses information from Chunk 7, then `depends_on: [11, 7]`

Important Notes:
- Make sure to include all dependencies for each chunk.
- Include both long-range and short-range dependencies.
- Do NOT forget about long-range dependencies.
- Try to be as comprehensive as possible.
- Make sure there is always a path from earlier chunks (e.g. problem_setup and/or active_computation) to the final answer.

---

### Output Format:

Return a single dictionary with one entry per chunk, where each entry has:
- the chunk index (as the key, converted to a string),
- a dictionary with:
    - `"function_tags"`: list of tag strings
    - `"depends_on"`: list of chunk indices, converted to strings

Here's the expected format:

```language=json
{{
    "4": {{
    "function_tags": ["plan_generation"],
    "depends_on": ["3"]
    }},
    "5": {{
    "function_tags": ["fact_retrieval"],
    "depends_on": []
    }},
    "9": {{
    "function_tags": ["active_computation"],
    "depends_on": ["4", "5"]
    }},
    "24": {{
    "function_tags": ["self_checking"],
    "depends_on": ["23"]
    }},
    "25": {{
    "function_tags": ["final_answer_emission"],
    "depends_on": ["23"]
    }}
}}
```

Here is the math problem:

[PROBLEM]
{problem_text}

Here is the full Chain of Thought, broken into chunks:

[CHUNKS]
{full_chunked_text}

Now label each chunk with function tags and dependencies.
"""

# ---------------------------------------------------------------------------
# Language-specific instruction templates
# ---------------------------------------------------------------------------

# MMMLU: instruct model to reason then write "Answer: X"
_MMMLU_SYSTEM_PROMPTS = {
    "en": "Answer this multiple-choice question step by step. Write your final answer on a new line as 'Answer: <letter>' where <letter> is A, B, C, or D.",
    "fr": "Répondez à cette question à choix multiples étape par étape. Écrivez votre réponse finale sur une nouvelle ligne sous la forme 'Réponse: <lettre>' où <lettre> est A, B, C ou D.",
    "zh": "请逐步回答这道多项选择题。在新的一行写出你的最终答案，格式为\u201c答案：<字母>\u201d，其中字母为A、B、C或D之一。",
    "ar": "أجب على هذا السؤال متعدد الخيارات خطوة بخطوة. اكتب إجابتك النهائية في سطر جديد بالشكل 'الإجابة: <حرف>' حيث يكون الحرف A أو B أو C أو D.",
    "de": "Beantworte diese Multiple-Choice-Frage Schritt für Schritt. Schreibe deine endgültige Antwort auf eine neue Zeile als 'Antwort: <Buchstabe>', wobei <Buchstabe> A, B, C oder D ist.",
    "es": "Responde esta pregunta de opción múltiple paso a paso. Escribe tu respuesta final en una nueva línea como 'Respuesta: <letra>' donde <letra> es A, B, C o D.",
    "hi": "इस बहुविकल्पीय प्रश्न का उत्तर चरण-दर-चरण दें। अपना अंतिम उत्तर एक नई पंक्ति पर 'उत्तर: <अक्षर>' के रूप में लिखें जहाँ <अक्षर> A, B, C या D है।",
    "bn": "এই বহু-নির্বাচনী প্রশ্নটি ধাপে ধাপে উত্তর দিন। আপনার চূড়ান্ত উত্তর একটি নতুন লাইনে 'উত্তর: <অক্ষর>' হিসেবে লিখুন যেখানে <অক্ষর> হল A, B, C বা D।",
    "id": "Jawab pertanyaan pilihan ganda ini langkah demi langkah. Tulis jawaban akhir Anda pada baris baru sebagai 'Jawaban: <huruf>' di mana <huruf> adalah A, B, C, atau D.",
    "it": "Rispondi a questa domanda a scelta multipla passo dopo passo. Scrivi la tua risposta finale su una nuova riga come 'Risposta: <lettera>' dove <lettera> è A, B, C o D.",
    "ja": "この多肢選択問題に段階的に答えてください。最終的な答えを新しい行に「答え：<文字>」の形式で書いてください（<文字>はA、B、C、またはDです）。",
    "ko": "이 객관식 문제에 단계별로 답하세요. 최종 답을 새 줄에 '정답: <글자>' 형식으로 적으세요. 여기서 <글자>는 A, B, C 또는 D입니다.",
    "pt": "Responda a esta questão de múltipla escolha passo a passo. Escreva sua resposta final em uma nova linha como 'Resposta: <letra>' onde <letra> é A, B, C ou D.",
    "sw": "Jibu swali hili la chaguo nyingi hatua kwa hatua. Andika jibu lako la mwisho kwenye mstari mpya kama 'Jibu: <herufi>' ambapo <herufi> ni A, B, C, au D.",
    "yo": "Answer this multiple-choice question step by step. Write your final answer on a new line as 'Answer: <letter>' where <letter> is A, B, C, or D.",
}

# MMATH: instruct model to use \boxed{} for answer
_MMATH_SYSTEM_PROMPTS = {
    "en": "Solve this math problem step by step. You MUST put your final answer in \\boxed{{}}.",
    "fr": "Résolvez ce problème de mathématiques étape par étape. Vous DEVEZ mettre votre réponse finale dans \\boxed{{}}.",
    "zh": "请逐步解决这个数学问题。你必须将最终答案放在 \\boxed{{}} 中。",
    "ar": "حل هذه المسألة الرياضية خطوة بخطوة. يجب أن تضع إجابتك النهائية في \\boxed{{}}.",
}

# ---------------------------------------------------------------------------
# Think-token language anchors
# Inserted immediately after <think> to prime the model to reason in the
# target language.
# ---------------------------------------------------------------------------

THINK_ANCHORS = {
    "en": "I will now start thinking in English.",
    "fr": "Je vais maintenant commencer à penser en français.",
    "zh": "我现在将开始用中文思考。",
    "ar": "سأبدأ الآن في التفكير باللغة العربية.",
    "de": "Ich werde jetzt auf Deutsch denken.",
    "es": "Voy a comenzar a pensar en español.",
    "hi": "मैं अब हिंदी में सोचना शुरू करूंगा।",
    "bn": "আমি এখন বাংলায় চিন্তা করা শুরু করব।",
    "id": "Saya akan mulai berpikir dalam bahasa Indonesia.",
    "it": "Inizierò ora a pensare in italiano.",
    "ja": "今から日本語で考え始めます。",
    "ko": "이제 한국어로 생각하기 시작하겠습니다.",
    "pt": "Vou agora começar a pensar em português.",
    "sw": "Nitaanza sasa kufikiria kwa Kiswahili.",
    "yo": "Emi yoo bẹrẹ lati ronu ni ede Yorùbá.",
}

# ---------------------------------------------------------------------------
# Per-language labels for "Problem:" and "Solution:" scaffold words
# ---------------------------------------------------------------------------

_PROBLEM_LABELS = {
    "ar": "المسألة",
    "bn": "সমস্যা",
    "de": "Aufgabe",
    "en": "Problem",
    "es": "Problema",
    "fr": "Problème",
    "hi": "प्रश्न",
    "id": "Soal",
    "it": "Problema",
    "ja": "問題",
    "ko": "문제",
    "pt": "Problema",
    "ru": "Задача",
    "sw": "Tatizo",
    "te": "సమస్య",
    "th": "โจทย์",
    "yo": "Ìṣòro",
    "zh": "问题",
}

_SOLUTION_LABELS = {
    "ar": "الحل",
    "bn": "সমাধান",
    "de": "Lösung",
    "en": "Solution",
    "es": "Solución",
    "fr": "Solution",
    "hi": "हल",
    "id": "Penyelesaian",
    "it": "Soluzione",
    "ja": "解答",
    "ko": "풀이",
    "pt": "Solução",
    "ru": "Решение",
    "sw": "Suluhisho",
    "te": "సమాధానం",
    "th": "คำตอบ",
    "yo": "Ìdáhùn",
    "zh": "解答",
}

# MGSM: per-language system prompts
_MGSM_SYSTEM_PROMPTS = {
    "ar": "دائمًا فكر باللغة العربية. حل مسألة الرياضيات التالية خطوة بخطوة. اكتب تفكيرك ضمن <think>...</think>. وأخيرًا ضع النتيجة النهائية داخل \\boxed{}.",
    "bn": "অনুগ্রহ করে সবসময় বাংলায় ভাবুন। ধাপে ধাপে নিচের গণিত সমস্যা সমাধান করুন। <think>...</think> এ যুক্তি লিখুন এবং চূড়ান্ত ফলাফল \\boxed{} এ দিন।",
    "de": "Bitte denken Sie immer auf Deutsch. Lösen Sie das folgende Mathematikproblem Schritt für Schritt. Schreiben Sie Ihre Begründung in <think>...</think>. Geben Sie schließlich das Ergebnis in \\boxed{} an.",
    "en": "Always think in English. Solve the following math problem step by step. Write your reasoning in <think>...</think>. Finally, provide the final result enclosed in \\boxed{}.",
    "es": "Piense siempre en español. Resuelva el siguiente problema de matemáticas paso a paso. Escriba su razonamiento en <think>...</think> y encierre el resultado final en \\boxed{}.",
    "fr": "Veuillez toujours réfléchir en français. Résolvez le problème mathématique suivant étape par étape. Écrivez le raisonnement dans <think>...</think>. Enfin, encadrez le résultat final dans \\boxed{}.",
    "hi": "कृपया हमेशा हिंदी में सोचें। नीचे दिए गए गणित प्रश्न को चरणबद्ध तरीके से हल करें। तर्क <think>...</think> में लिखें और अंतिम परिणाम \\boxed{} में दें।",
    "id": "Selalu berpikir dalam bahasa Indonesia. Selesaikan soal matematika berikut langkah demi langkah. Tulis penalaran di <think>...</think> dan berikan hasil akhir dalam \\boxed{}.",
    "it": "Pensa sempre in italiano. Risolvi il seguente problema matematico passo dopo passo. Scrivi il ragionamento in <think>...</think>. Infine racchiudi il risultato in \\boxed{}.",
    "ja": "常に日本語で考えてください。以下の数学問題を段階的に解いてください。推論は <think>...</think> に記述し、最終結果を \\boxed{} に示してください。",
    "ko": "항상 한국어로 사고하세요. 다음 수학 문제를 단계별로 풀이하세요. 추론은 <think>...</think>에 쓰고 최종 결과를 제시하세요.",
    "pt": "A pedido, pense sempre em português. Resolva o problema de matemática a seguir passo a passo. Escreva o raciocínio em <think>...</think> e coloque o resultado final em \\boxed{}.",
    "ru": "Всегда рассуждай по-русски. Решай следующую математическую задачу шаг за шагом. Пиши рассуждения внутри <think>...</think>. В конце помести итог внутри \\boxed{}.",
    "sw": "Tafadhali kila mara fikiria kwa Kiswahili. Tatua tatizo lifuatalo la hisabati hatua kwa hatua. Andika hoja kwenye <think>...</think> na weka matokeo ya mwisho ndani ya \\boxed{}.",
    "te": "దయచేసి ఎల్లప్పుదూ తెలుగులో ఆలోచించండి. క్రింది గణిత సమస్యను దశలవారీగా పరిష్కరించండి. మీ తర్కాన్ని <think>...</think> లో రాయండి. చివరగా తుది ఫలితాన్ని \\boxed{} లో ఇవ్వండి.",
    "th": "โปรดคิดเป็นภาษาไทยเสมอ แก้ปัญหาคณิตต่อไปนี้แบบเป็นขั้นตอน เขียนเหตุผลไว้ใน <think>...</think> และใส่ผลลัพธ์สุดท้ายใน \\boxed{}.",
    "yo": "Jòwó máa rò ní Yorùbá. Şe işoro işirò yíí ní igbèsè-nípèyà. Kọ irònú sínú <think>...</think> kí o sì fi esi ikẹhin sínú \\boxed{}.",
    "zh": "请始终用中文思考。逐步解决以下数学问题。每一步将推理写在 <think>...</think> 中。最后，请将最终结果放在 \\boxed{} 中。",
}


def build_base_solution_prompt(problem: Problem, language: str) -> str:
    """
    Build the prompt for generating a base solution.

    Args:
        problem: The Problem to solve
        language: Language code for the prompt

    Returns:
        Full prompt string ready for API call
    """
    if problem.answer_type == "latex_boxed":
        system = _MMATH_SYSTEM_PROMPTS.get(language, _MMATH_SYSTEM_PROMPTS["en"])
    elif problem.answer_type == "multiple_choice":
        system = _MMMLU_SYSTEM_PROMPTS.get(language, _MMMLU_SYSTEM_PROMPTS["en"])
    else:
        system = _MGSM_SYSTEM_PROMPTS.get(language, _MGSM_SYSTEM_PROMPTS["en"])

    problem_label = _PROBLEM_LABELS.get(language, _PROBLEM_LABELS["en"])
    solution_label = _SOLUTION_LABELS.get(language, _SOLUTION_LABELS["en"])
    think_anchor = THINK_ANCHORS.get(language, "")
    anchor_text = f"{think_anchor}\n" if think_anchor else ""
    prompt = f"{system} {problem_label}: {problem.question} {solution_label}: \n<think>\n{anchor_text}"
    return prompt


def build_rollout_prompt(
    problem: Problem,
    prefix_without_chunk: str,
    language: str,
    rollout_type: str = "default",
) -> str:
    """
    Build the prompt for a rollout (continuation after chunk removal).

    Args:
        problem: The Problem being solved
        prefix_without_chunk: The CoT prefix with the target chunk removed
        language: Language code
        rollout_type: "default" or "forced_answer"

    Returns:
        Full prompt string ready for API call
    """
    if problem.answer_type == "latex_boxed":
        system = _MMATH_SYSTEM_PROMPTS.get(language, _MMATH_SYSTEM_PROMPTS["en"])
    elif problem.answer_type == "multiple_choice":
        system = _MMMLU_SYSTEM_PROMPTS.get(language, _MMMLU_SYSTEM_PROMPTS["en"])
    else:
        system = _MGSM_SYSTEM_PROMPTS.get(language, _MGSM_SYSTEM_PROMPTS["en"])

    problem_label = _PROBLEM_LABELS.get(language, _PROBLEM_LABELS["en"])
    solution_label = _SOLUTION_LABELS.get(language, _SOLUTION_LABELS["en"])
    think_anchor = THINK_ANCHORS.get(language, "")
    anchor_text = f"{think_anchor}\n" if think_anchor else ""
    prompt = f"{system} {problem_label}: {problem.question} {solution_label}: \n<think>\n{anchor_text}{prefix_without_chunk}"

    if rollout_type == "forced_answer":
        if problem.answer_type == "latex_boxed":
            prompt += "\n</think>\n\nTherefore, the final answers is \\boxed{"
        elif problem.answer_type == "multiple_choice":
            prompt += "\n</think>\n\nAnswer: "
        else:
            # For MGSM, force a language-specific "Final:" marker
            if language == "zh":
                prompt += "\n</think>\n\n最终答案："
            elif language == "ar":
                prompt += "\n</think>\n\nالإجابة النهائية: "
            elif language == "fr":
                prompt += "\n</think>\n\nRéponse finale: "
            else:
                prompt += "\n</think>\n\nFinal: "

    return prompt


def build_dag_labeling_prompt(
    problem_text: str,
    chunks: list[str],
    language: str,
) -> str:
    """
    Build the prompt for GPT-4o DAG taxonomy labeling.

    Wraps DAG_PROMPT with chunk formatting. Adds language context
    for non-English CoTs. Always requests English output for labels.

    Args:
        problem_text: The problem statement
        chunks: List of chunk strings
        language: Language code of the CoT

    Returns:
        Formatted prompt string
    """
    full_chunked_text = ""
    for i, chunk in enumerate(chunks):
        full_chunked_text += f"Chunk {i}:\n{chunk}\n\n"

    formatted_prompt = DAG_PROMPT.format(
        problem_text=problem_text,
        full_chunked_text=full_chunked_text,
    )

    # Add language context for non-English CoTs
    if language != "en":
        lang_name = LANGUAGE_NAMES.get(language, language)
        language_note = (
            f"\n\nNote: The Chain of Thought above is written in {lang_name}. "
            f"Please still output all function_tags and depends_on in English as specified above."
        )
        formatted_prompt += language_note

    return formatted_prompt

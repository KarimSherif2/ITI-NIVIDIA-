"""
guardrails.py
Input safety and output grounding guardrails.
Note: Relevance check removed - LLM handles off-topic questions gracefully.
"""

BLOCKED_PHRASES = [
    "how to hack", "make a bomb", "illegal", "steal", "fraud",
    "bypass security", "jailbreak", "ignore previous instructions"
]


def check_input(question: str) -> tuple:
    """
    Returns (is_safe, reason).
    is_safe=False means the question should be blocked.
    """
    q_lower = question.lower().strip()

    if len(q_lower) < 3:
        return False, "Question is too short. Please ask a meaningful question."

    for phrase in BLOCKED_PHRASES:
        if phrase in q_lower:
            return False, (
                "Your question contains content outside the scope of document analysis. "
                "Please ask questions related to the uploaded document."
            )

    return True, ""


def check_output(answer: str, source_docs: list) -> tuple:
    """
    Returns (final_answer, was_flagged).
    Appends a disclaimer if the answer has no supporting sources.
    """
    not_found_signals = [
        "i could not find",
        "not mentioned",
        "not found in",
        "no information",
        "not specified in"
    ]

    # If LLM already says not found, trust it
    if any(s in answer.lower() for s in not_found_signals):
        return answer, False

    # If no source docs were retrieved, flag the answer
    if not source_docs:
        disclaimer = (
            "\n\n⚠️ **Warning:** No relevant passages were found in the document "
            "to support this answer. Please verify independently."
        )
        return answer + disclaimer, True

    return answer, False

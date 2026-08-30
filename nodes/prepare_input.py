"""
prepare_input.py - Input Preparation Node

Routes between query mode (generate answer) and external mode (check pasted text).
"""

from typing import Any, Dict


def prepare_input_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Determine whether to generate an answer or use external text.
    """
    external_answer = state.get("external_answer", "")
    query = state.get("query", "")

    if external_answer and external_answer.strip():
        # External mode - use provided text
        return {
            "answer": external_answer.strip(),
            "concepts": [],
            "answer_source": "external",
            "query": query if query else "External text check",
        }
    else:
        # Generation mode
        return {
            "answer_source": "generated",
        }


def should_generate_answer(state: Dict[str, Any]) -> str:
    """
    Conditional routing function.
    Returns "claims" to skip answer generation, or "answer" to generate.
    """
    external_answer = state.get("external_answer", "")

    if external_answer and external_answer.strip():
        return "claims"  # Skip answer generation
    else:
        return "answer"  # Generate answer first

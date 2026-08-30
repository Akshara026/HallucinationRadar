"""
state.py - RadarState definition for LangGraph pipeline
"""

from typing import TypedDict, List, Dict, Any, Optional


class RadarState(TypedDict, total=False):
    """State that flows through the HallucinationRadar pipeline."""

    # Input - EITHER query (generate answer) OR external_answer (check pasted text)
    query: str
    external_answer: str  # if provided, skips answer_node entirely

    # Set by prepare_input_node
    answer_source: str  # "generated" or "external"

    # Answer node output (or copied from external_answer)
    concepts: List[str]
    answer: str

    # Claims node output
    claims: List[str]

    # Retrieval node output
    evidence: Dict[str, List[Dict[str, Any]]]

    # Verify node output
    verdicts: Dict[str, Dict[str, Any]]

    # Score node output
    score: Dict[str, Any]

    # Report node output
    report: str

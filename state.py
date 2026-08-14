"""
state.py - RadarState definition for LangGraph pipeline
"""

from typing import TypedDict, List, Dict, Any


class RadarState(TypedDict, total=False):
    """State that flows through the HallucinationRadar pipeline."""

    # Input
    query: str

    # Answer node output
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

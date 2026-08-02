"""
score.py - Hallucination Scoring Node

Pure math/logic node - no LLM calls needed.
Takes verdicts from verify.py and produces:
- Overall hallucination score (0-1)
- Verdict label
- Hallucination risk level
- Claim-level breakdown
- High-risk and verified claim lists
"""

from typing import Any, Dict, List

# Verdict weight mapping
VERDICT_WEIGHTS = {
    "SUPPORTED": 1.0,
    "CONTRADICTED": 0.0,
    "INSUFFICIENT_EVIDENCE": 0.4,
    "UNVERIFIABLE": 0.5,
}


def score_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate hallucination score from verification verdicts.

    Input:  state["verdicts"] - dict mapping claim → verification result
    Output: state["score"] - dict with overall score and breakdown
    """
    verdicts = state.get("verdicts", {})

    if not verdicts:
        return {
            "score": {
                "overall_score": 0.0,
                "verdict": "NO_CLAIMS",
                "hallucination_risk": "UNKNOWN",
                "breakdown": {
                    "total": 0,
                    "supported": 0,
                    "contradicted": 0,
                    "unverifiable": 0,
                    "insufficient_evidence": 0,
                },
                "high_risk_claims": [],
                "verified_claims": [],
            }
        }

    # 1. Calculate overall score
    overall_score = calculate_weighted_score(verdicts)

    # 2. Get breakdown counts
    breakdown = get_breakdown(verdicts)

    # 3. Identify high risk claims (contradicted with confidence > 60%)
    high_risk = get_high_risk_claims(verdicts, confidence_threshold=0.6)

    # 4. Get verified claims (supported with confidence > 70%)
    verified = get_verified_claims(verdicts, confidence_threshold=0.7)

    # 5. Risk level
    risk = get_risk_level(overall_score, breakdown["contradicted"])

    # 6. Verdict label
    label = get_verdict_label(overall_score)

    print(f"\n📊 Score calculated: {overall_score:.3f} ({label})")
    print(f"   Risk: {risk}")
    print(
        f"   Supported: {breakdown['supported']} | "
        f"Contradicted: {breakdown['contradicted']} | "
        f"Unverifiable: {breakdown['unverifiable']} | "
        f"Insufficient: {breakdown['insufficient_evidence']}"
    )

    if high_risk:
        print(f"   ⚠️  High risk claims: {len(high_risk)}")

    return {
        "score": {
            "overall_score": overall_score,
            "verdict": label,
            "hallucination_risk": risk,
            "breakdown": breakdown,
            "high_risk_claims": high_risk,
            "verified_claims": verified,
        }
    }


def calculate_weighted_score(verdicts: Dict[str, Dict[str, Any]]) -> float:
    """
    Calculate overall hallucination score.

    Formula:
    - Each claim contributes: base_weight * confidence
    - Final score = weighted_sum / total_weight

    This ensures:
    - High confidence SUPPORTED claims pull score up
    - High confidence CONTRADICTED claims pull score down
    - Low confidence verdicts have less impact
    """
    if not verdicts:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0

    for claim, verdict_data in verdicts.items():
        verdict = verdict_data.get("verdict", "UNVERIFIABLE")
        confidence = verdict_data.get("confidence", 0.5)

        # Get base score for this verdict type
        base_score = VERDICT_WEIGHTS.get(verdict, 0.5)

        # Weight by confidence
        # High confidence verdicts matter more
        # Minimum weight of 0.3 for zero-confidence verdicts
        weight = max(confidence, 0.3) if confidence > 0 else 0.3

        weighted_sum += base_score * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return round(weighted_sum / total_weight, 3)


def get_breakdown(verdicts: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """
    Count verdicts by type.
    """
    breakdown = {
        "total": len(verdicts),
        "supported": 0,
        "contradicted": 0,
        "unverifiable": 0,
        "insufficient_evidence": 0,
    }

    for verdict_data in verdicts.values():
        verdict = verdict_data.get("verdict", "UNVERIFIABLE")

        if verdict == "SUPPORTED":
            breakdown["supported"] += 1
        elif verdict == "CONTRADICTED":
            breakdown["contradicted"] += 1
        elif verdict == "UNVERIFIABLE":
            breakdown["unverifiable"] += 1
        elif verdict == "INSUFFICIENT_EVIDENCE":
            breakdown["insufficient_evidence"] += 1
        else:
            # Unknown verdict type, count as unverifiable
            breakdown["unverifiable"] += 1

    return breakdown


def get_high_risk_claims(
    verdicts: Dict[str, Dict[str, Any]], confidence_threshold: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Identify claims that are contradicted with high confidence.
    These are the strongest hallucination signals.
    """
    high_risk = []

    for claim, verdict_data in verdicts.items():
        verdict = verdict_data.get("verdict", "")
        confidence = verdict_data.get("confidence", 0)

        if verdict == "CONTRADICTED" and confidence >= confidence_threshold:
            high_risk.append(
                {
                    "claim": claim,
                    "confidence": confidence,
                    "reasoning": verdict_data.get("reasoning", "")[:200],
                    "discrepancies": verdict_data.get("discrepancies", []),
                }
            )

    high_risk.sort(key=lambda x: x["confidence"], reverse=True)

    return high_risk


def get_verified_claims(
    verdicts: Dict[str, Dict[str, Any]], confidence_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Identify claims that are supported with high confidence.
    These are the most reliable claims.
    """
    verified = []

    for claim, verdict_data in verdicts.items():
        verdict = verdict_data.get("verdict", "")
        confidence = verdict_data.get("confidence", 0)

        if verdict == "SUPPORTED" and confidence >= confidence_threshold:
            verified.append(
                {
                    "claim": claim,
                    "confidence": confidence,
                    "evidence_used": verdict_data.get("evidence_used", []),
                    "key_facts_matched": verdict_data.get("key_facts_matched", []),
                }
            )

    verified.sort(key=lambda x: x["confidence"], reverse=True)

    return verified


def get_risk_level(score: float, contradicted_count: int) -> str:
    """
    Determine hallucination risk level.

    Risk factors:
    - Multiple contradictions = HIGH risk regardless of score
    - Low score = HIGH risk
    - Medium score = MEDIUM risk
    - High score = LOW risk (unless contradictions exist)
    """
    # Even one high-confidence contradiction is concerning
    # Two or more = definitely HIGH risk
    if contradicted_count >= 2:
        return "HIGH"

    if contradicted_count == 1 and score < 0.5:
        return "HIGH"

    if score >= 0.80:
        return "LOW"

    if score >= 0.60:
        return "MEDIUM"

    return "HIGH"


def get_verdict_label(score: float) -> str:
    """
    Human-readable verdict based on score.
    """
    if score >= 0.85:
        return "WELL_SUPPORTED"
    if score >= 0.65:
        return "PARTIALLY_SUPPORTED"
    if score >= 0.45:
        return "MOSTLY_UNVERIFIED"
    return "LIKELY_HALLUCINATED"

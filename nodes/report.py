"""
report.py - Report Generation Node

Generates a human-readable hallucination report.
Pure formatting - no LLM calls needed.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional


def report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a complete hallucination radar report.

    Input:  Full state dict with query, answer, claims, verdicts, score
    Output: state["report"] - formatted report string
    """
    query = state.get("query", "Unknown")
    answer = state.get("answer", "")
    claims = state.get("claims", [])
    verdicts = state.get("verdicts", {})
    score_data = state.get("score", {})

    report = generate_report(query, answer, claims, verdicts, score_data)

    print(f"📄 Report generated ({len(report)} characters)")

    return {"report": report}


def generate_report(
    query: str,
    answer: str,
    claims: List[str],
    verdicts: Dict[str, Any],
    score_data: Dict[str, Any],
) -> str:
    """Build the complete report string."""

    lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ═══════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════
    lines.append("╔══════════════════════════════════════════════════════════════╗")
    lines.append("║              HALLUCINATION RADAR REPORT                      ║")
    lines.append("╚══════════════════════════════════════════════════════════════╝")
    lines.append("")
    lines.append(f'Query:     "{query}"')
    lines.append(f"Generated: {timestamp}")
    lines.append("")

    # ═══════════════════════════════════════════
    # OVERALL VERDICT
    # ═══════════════════════════════════════════
    overall_score = score_data.get("overall_score", 0.0)
    verdict_label = score_data.get("verdict", "UNKNOWN")
    risk_level = score_data.get("hallucination_risk", "UNKNOWN")

    if overall_score >= 0.80:
        indicator = "🟢"
    elif overall_score >= 0.60:
        indicator = "🟡"
    elif overall_score >= 0.40:
        indicator = "🟠"
    else:
        indicator = "🔴"

    lines.append("┌──────────────────────────────────────────────────────────────┐")
    lines.append(f"│  OVERALL VERDICT                                             │")
    lines.append(f"│                                                              │")
    lines.append(f"│  {indicator} {verdict_label:<46} │")
    lines.append(f"│                                                              │")
    lines.append(
        f"│  Hallucination Score: {overall_score:.3f} / 1.000                            │"
    )
    lines.append(f"│  Risk Level:         {risk_level:<41} │")
    lines.append("└──────────────────────────────────────────────────────────────┘")
    lines.append("")

    # ═══════════════════════════════════════════
    # SCORE BREAKDOWN
    # ═══════════════════════════════════════════
    breakdown = score_data.get("breakdown", {})
    total = breakdown.get("total", len(claims))
    supported = breakdown.get("supported", 0)
    contradicted = breakdown.get("contradicted", 0)
    unverifiable = breakdown.get("unverifiable", 0)
    insufficient = breakdown.get("insufficient_evidence", 0)

    lines.append("┌──────────────────────────────────────────────────────────────┐")
    lines.append(f"│  CLAIM BREAKDOWN                                             │")
    lines.append(f"│                                                              │")
    lines.append(f"│  Total Claims Analyzed: {total:<36} │")
    lines.append(f"│                                                              │")

    if total > 0:
        lines.append(
            f"│  ✅ Supported:            {supported:<4} ({supported / total * 100:5.1f}%)                │"
        )
        lines.append(
            f"│  ❌ Contradicted:         {contradicted:<4} ({contradicted / total * 100:5.1f}%)                │"
        )
        lines.append(
            f"│  ❓ Unverifiable:         {unverifiable:<4} ({unverifiable / total * 100:5.1f}%)                │"
        )
        lines.append(
            f"│  ⚠️  Insufficient Evidence: {insufficient:<4} ({insufficient / total * 100:5.1f}%)                │"
        )
    else:
        lines.append(f"│  No claims analyzed                                         │")

    lines.append("└──────────────────────────────────────────────────────────────┘")
    lines.append("")

    # ═══════════════════════════════════════════
    # HIGH RISK CLAIMS
    # ═══════════════════════════════════════════
    high_risk = score_data.get("high_risk_claims", [])

    if high_risk:
        lines.append("┌──────────────────────────────────────────────────────────────┐")
        lines.append(
            f"│  ⚠️  LIKELY HALLUCINATIONS ({len(high_risk)} found)                               │"
        )
        lines.append("└──────────────────────────────────────────────────────────────┘")
        lines.append("")

        for i, claim_data in enumerate(high_risk, 1):
            claim_text = claim_data.get("claim", "Unknown claim")
            confidence = claim_data.get("confidence", 0)
            reasoning = claim_data.get("reasoning", "No reasoning provided")
            discrepancies = claim_data.get("discrepancies", [])

            lines.append(f"  ❌ #{i}: {wrap_text(claim_text, 60)}")
            lines.append(f"     Confidence: {confidence:.1%}")

            if discrepancies:
                lines.append(f"     Discrepancies:")
                for d in discrepancies[:3]:
                    lines.append(f"       • {wrap_text(str(d), 56)}")

            lines.append(f"     Reason: {wrap_text(reasoning[:200], 56)}")
            lines.append("")
    else:
        lines.append("  ✅ No likely hallucinations detected.")
        lines.append("")

    # ═══════════════════════════════════════════
    # VERIFIED CLAIMS
    # ═══════════════════════════════════════════
    verified = score_data.get("verified_claims", [])

    if verified:
        lines.append("┌──────────────────────────────────────────────────────────────┐")
        lines.append(
            f"│  ✅ VERIFIED CLAIMS ({len(verified)} confirmed)                               │"
        )
        lines.append("└──────────────────────────────────────────────────────────────┘")
        lines.append("")

        for i, claim_data in enumerate(verified, 1):
            claim_text = claim_data.get("claim", "Unknown claim")
            confidence = claim_data.get("confidence", 0)
            evidence = claim_data.get("evidence_used", [])

            lines.append(f"  ✅ #{i}: {wrap_text(claim_text, 60)}")
            lines.append(f"     Confidence: {confidence:.1%}")
            if evidence:
                lines.append(f"     Sources: {', '.join(evidence[:3])}")
            lines.append("")
    else:
        lines.append("  ⚠️  No claims could be verified with high confidence.")
        lines.append("")

    # ═══════════════════════════════════════════
    # ANNOTATED ANSWER
    # ═══════════════════════════════════════════
    if answer and claims:
        lines.append("┌──────────────────────────────────────────────────────────────┐")
        lines.append("│  ANNOTATED ANSWER                                            │")
        lines.append("│  ✅ = Supported  ❌ = Contradicted  ❓ = Unverified          │")
        lines.append("└──────────────────────────────────────────────────────────────┘")
        lines.append("")

        annotated = annotate_answer(answer, claims, verdicts)
        lines.append(annotated)
        lines.append("")

    # ═══════════════════════════════════════════
    # EVIDENCE SOURCES
    # ═══════════════════════════════════════════
    all_sources = collect_all_sources(verdicts)

    if all_sources:
        lines.append("┌──────────────────────────────────────────────────────────────┐")
        lines.append("│  EVIDENCE SOURCES                                            │")
        lines.append("└──────────────────────────────────────────────────────────────┘")
        lines.append("")

        for source in all_sources:
            lines.append(f"  📚 {source}")
        lines.append("")

    # ═══════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════
    lines.append("─" * 66)
    lines.append(f"Report generated by HallucinationRadar on {timestamp}")
    lines.append("Evidence sourced from Wikipedia")
    lines.append("─" * 66)

    return "\n".join(lines)


def wrap_text(text: str, width: int) -> str:
    """Wrap text to specified width with proper indentation on continuation lines."""
    if len(text) <= width:
        return text

    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > width:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1

    if current_line:
        lines.append(" ".join(current_line))

    return (
        "\n" + "\n".join(f"       {line}" for line in lines)
        if len(lines) > 1
        else lines[0]
    )


def annotate_answer(answer: str, claims: List[str], verdicts: Dict[str, Any]) -> str:
    """
    Annotate each sentence in the answer with its verification status.
    Uses fuzzy matching to map claims back to original sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return answer

    annotated_lines = []

    for sentence in sentences:
        best_match = find_matching_claim(sentence, claims, verdicts)

        if best_match:
            verdict = best_match.get("verdict", "UNVERIFIABLE")
            confidence = best_match.get("confidence", 0)

            if verdict == "SUPPORTED" and confidence >= 0.7:
                indicator = "✅"
            elif verdict == "SUPPORTED":
                indicator = "🟢"
            elif verdict == "CONTRADICTED":
                indicator = "❌"
            elif verdict == "INSUFFICIENT_EVIDENCE":
                indicator = "⚠️"
            else:
                indicator = "❓"
        else:
            indicator = "❓"

        annotated_lines.append(f"  {indicator} {sentence}")

    return "\n".join(annotated_lines)


def find_matching_claim(
    sentence: str, claims: List[str], verdicts: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Find the claim that best matches a sentence using word overlap.
    Returns the verdict data for the best matching claim.
    """
    sent_words = set(sentence.lower().split())

    if not sent_words:
        return None

    best_claim = None
    best_score = 0.0

    for claim in claims:
        claim_words = set(claim.lower().split())

        if not claim_words:
            continue

        overlap = len(sent_words & claim_words)
        union = len(sent_words | claim_words)

        if union == 0:
            continue

        score = overlap / union

        if score > best_score and score > 0.3:
            best_score = score
            best_claim = claim

    if best_claim and best_claim in verdicts:
        return verdicts[best_claim]

    return None


def collect_all_sources(verdicts: Dict[str, Any]) -> List[str]:
    """
    Collect all unique evidence sources used across all verdicts.
    """
    sources = set()

    for verdict_data in verdicts.values():
        evidence_used = verdict_data.get("evidence_used", [])
        for source in evidence_used:
            if source and source != "No evidence found" and source != "Error":
                sources.add(source)

    return sorted(sources)


def save_report(report: str, filepath: Optional[str] = None) -> str:
    """
    Save report to file. If no path given, saves to data/reports/ with timestamp.
    """
    import os

    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("data/reports", exist_ok=True)
        filepath = f"data/reports/report_{timestamp}.txt"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    print(f" Report saved to: {filepath}")
    return filepath

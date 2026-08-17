"""
report.py - Report Generation Node

Generates a human-readable hallucination report.
Pure formatting - no LLM calls needed.
"""

import re
import os
from typing import Dict, List, Any, Optional
from datetime import datetime


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

    report_text = generate_report(query, answer, claims, verdicts, score_data)

    print(f"📄 Report generated ({len(report_text)} characters)")

    return {"report": report_text}


def generate_report(
    query: str,
    answer: str,
    claims: List[str],
    verdicts: Dict[str, Any],
    score_data: Dict[str, Any]
) -> str:
    """Build the complete report string."""

    lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # HEADER
    lines.append("╔══════════════════════════════════════════════════════════════╗")
    lines.append("║              HALLUCINATION RADAR REPORT                      ║")
    lines.append("╚══════════════════════════════════════════════════════════════╝")
    lines.append("")
    lines.append(f"Query:     \"{query}\"")
    lines.append(f"Generated: {timestamp}")
    lines.append("")

    # OVERALL VERDICT
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
    lines.append(f"│  Hallucination Score: {overall_score:.3f} / 1.000                            │")
    lines.append(f"│  Risk Level:         {risk_level:<41} │")
    lines.append("└──────────────────────────────────────────────────────────────┘")
    lines.append("")

    # SCORE BREAKDOWN
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
        lines.append(f"│  ✅ Supported:            {supported:<4} ({supported/total*100:5.1f}%)                │")
        lines.append(f"│  ❌ Contradicted:         {contradicted:<4} ({contradicted/total*100:5.1f}%)                │")
        lines.append(f"│  ❓ Unverifiable:         {unverifiable:<4} ({unverifiable/total*100:5.1f}%)                │")
        lines.append(f"│  ⚠️  Insufficient Evidence: {insufficient:<4} ({insufficient/total*100:5.1f}%)                │")
    else:
        lines.append(f"│  No claims analyzed                                         │")

    lines.append("└──────────────────────────────────────────────────────────────┘")
    lines.append("")

    # HIGH RISK CLAIMS
    high_risk = score_data.get("high_risk_claims", [])

    if high_risk:
        lines.append("┌──────────────────────────────────────────────────────────────┐")
        lines.append(f"│  ⚠️  LIKELY HALLUCINATIONS ({len(high_risk)} found)                               │")
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

    # VERIFIED CLAIMS
    verified = score_data.get("verified_claims", [])

    if verified:
        lines.append("┌──────────────────────────────────────────────────────────────┐")
        lines.append(f"│  ✅ VERIFIED CLAIMS ({len(verified)} confirmed)                               │")
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

    # ANNOTATED ANSWER - IMPROVED WITH CONTRADICTION DETECTION
    if answer and claims:
        lines.append("┌──────────────────────────────────────────────────────────────┐")
        lines.append("│  ANNOTATED ANSWER                                            │")
        lines.append("│  ✅ = Supported  ❌ = Contradicted  ⚠️ = Mixed  ❓ = Unverified│")
        lines.append("└──────────────────────────────────────────────────────────────┘")
        lines.append("")

        annotated = annotate_answer_with_mixed_detection(answer, claims, verdicts)
        lines.append(annotated)
        lines.append("")

    # EVIDENCE SOURCES
    all_sources = collect_relevant_sources(verdicts)

    if all_sources:
        lines.append("┌──────────────────────────────────────────────────────────────┐")
        lines.append("│  EVIDENCE SOURCES                                            │")
        lines.append("└──────────────────────────────────────────────────────────────┘")
        lines.append("")

        for source in all_sources:
            lines.append(f"  📚 {source}")
        lines.append("")

    # FOOTER
    lines.append("─" * 66)
    lines.append(f"Report generated by HallucinationRadar on {timestamp}")
    lines.append("Evidence sourced from Wikipedia")
    lines.append("─" * 66)

    return "\n".join(lines)


def annotate_answer_with_mixed_detection(
    answer: str,
    claims: List[str],
    verdicts: Dict[str, Any]
) -> str:
    """
    Annotate each sentence with its verification status.
    Detects sentences that contain MULTIPLE claims with different verdicts.
    Marks as ⚠️ MIXED if sentence has both supported and contradicted claims.
    """
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return answer

    annotated_lines = []

    for sentence in sentences:
        # Find ALL claims that match this sentence
        matching_claims = find_all_matching_claims(sentence, claims, verdicts)

        if not matching_claims:
            annotated_lines.append(f"  ❓ {sentence}")
            continue

        # Check if there are multiple verdicts for this sentence
        verdicts_in_sentence = set()
        has_contradiction = False
        has_support = False

        for claim_data in matching_claims:
            verdict = claim_data.get("verdict", "UNVERIFIABLE")
            verdicts_in_sentence.add(verdict)

            if verdict == "CONTRADICTED":
                has_contradiction = True
            elif verdict == "SUPPORTED":
                has_support = True

        # Determine indicator
        if has_contradiction and has_support:
            # MIXED - both true and false facts in this sentence
            indicator = "⚠️"
            annotation = " MIXED"
        elif has_contradiction:
            indicator = "❌"
            annotation = ""
        elif has_support:
            # Check confidence of supported claims
            supported_claims = [c for c in matching_claims if c.get("verdict") == "SUPPORTED"]
            max_confidence = max((c.get("confidence", 0) for c in supported_claims), default=0)
            indicator = "✅" if max_confidence >= 0.7 else "🟢"
            annotation = ""
        else:
            indicator = "❓"
            annotation = ""

        annotated_lines.append(f"  {indicator}{annotation} {sentence}")

    return "\n".join(annotated_lines)


def find_all_matching_claims(
    sentence: str,
    claims: List[str],
    verdicts: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Find ALL claims that match a sentence (not just the best one).
    Returns list of verdict data for matching claims.
    """
    sent_words = set(sentence.lower().split())

    if not sent_words:
        return []

    matching_claims = []

    for claim in claims:
        claim_words = set(claim.lower().split())

        if not claim_words:
            continue

        overlap = len(sent_words & claim_words)
        union = len(sent_words | claim_words)

        if union == 0:
            continue

        score = overlap / union

        # Lower threshold to catch partial matches
        if score > 0.15:  # Was 0.3, now more lenient
            if claim in verdicts:
                matching_claims.append(verdicts[claim])

    return matching_claims


def wrap_text(text: str, width: int) -> str:
    """Wrap text to specified width with proper indentation."""
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

    return "\n" + "\n".join(f"       {line}" for line in lines) if len(lines) > 1 else lines[0]


def collect_relevant_sources(verdicts: Dict[str, Any]) -> List[str]:
    """
    Collect only relevant sources from SUPPORTED and CONTRADICTED claims.
    Filters out sources from UNVERIFIABLE and INSUFFICIENT_EVIDENCE claims.
    """
    sources = set()

    for verdict_data in verdicts.values():
        verdict = verdict_data.get("verdict", "")

        # Only include sources from meaningful verdicts
        if verdict in ["SUPPORTED", "CONTRADICTED"]:
            evidence_used = verdict_data.get("evidence_used", [])
            for source in evidence_used:
                if source and source != "No evidence found" and source != "Error":
                    sources.add(source)

    return sorted(sources)


def save_report_pdf(report_text: str, filepath: Optional[str] = None) -> str:
    """Save report as PDF using fpdf2."""
    from fpdf import FPDF

    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("data/reports", exist_ok=True)
        filepath = f"data/reports/report_{timestamp}.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Courier", size=8)

    for line in report_text.split("\n"):
        line_ascii = line
        line_ascii = line_ascii.replace('╔', '+').replace('╗', '+').replace('╚', '+').replace('╝', '+')
        line_ascii = line_ascii.replace('║', '|').replace('─', '-').replace('├', '+').replace('┤', '+')
        line_ascii = line_ascii.replace('┌', '+').replace('┐', '+').replace('└', '+').replace('┘', '+')
        line_ascii = line_ascii.replace('│', '|').replace('┬', '+').replace('┴', '+').replace('┼', '+')
        line_ascii = re.sub(r'[^\x00-\x7F]+', '', line_ascii)
        line_ascii = line_ascii.encode('latin-1', errors='replace').decode('latin-1')
        pdf.cell(0, 4, line_ascii, ln=True)

    pdf.output(filepath)
    print(f"💾 PDF report saved to: {filepath}")
    return filepath

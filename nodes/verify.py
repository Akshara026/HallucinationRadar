"""
verify.py - Claim Verification Node

Verifies each claim against retrieved Wikipedia evidence.
Single LLM call per claim (combines relevance check + verification).
Handles evidence quality with confidence capping.
Strict date/number matching to prevent false SUPPORTED verdicts.
"""

import json
import re
import time
from collections import Counter
from typing import Any, Dict, List

from langchain_ollama import ChatOllama

# Deterministic model for verification
llm = ChatOllama(model="qwen2.5:7b", temperature=0, num_ctx=4096)

# Valid verdicts for normalization
VALID_VERDICTS = {"SUPPORTED", "CONTRADICTED", "UNVERIFIABLE", "INSUFFICIENT_EVIDENCE"}


def normalize_verdict(verdict: str) -> str:
    """Fix common misspellings and invalid verdicts."""
    verdict = verdict.upper().strip()

    # Fix common misspellings
    if "CONTRADICT" in verdict or "CONTRADADICT" in verdict:
        return "CONTRADICTED"
    if "SUPPORT" in verdict:
        return "SUPPORTED"
    if "UNVERIF" in verdict:
        return "UNVERIFIABLE"
    if "INSUFFICIENT" in verdict or "INSUFFICENT" in verdict:
        return "INSUFFICIENT_EVIDENCE"
    if "IRRELEVANT" in verdict:
        return "UNVERIFIABLE"  # Map IRRELEVANT to UNVERIFIABLE

    if verdict not in VALID_VERDICTS:
        return "INSUFFICIENT_EVIDENCE"  # Safe default

    return verdict


def verify_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify all claims against their retrieved evidence.

    Input:  state["claims"] - list of claims
            state["evidence"] - dict mapping claim → list of evidence dicts

    Output: state["verdicts"] - dict mapping claim → verification result
    """
    claims = state.get("claims", [])
    evidence = state.get("evidence", {})

    if not claims:
        return {"verdicts": {}}

    print(f"\n🔬 Verifying {len(claims)} claims...")
    start_time = time.time()

    verdicts = {}

    for i, claim in enumerate(claims):
        evidence_list = evidence.get(claim, [])
        print(f"  {i + 1}/{len(claims)}: {claim[:80]}...")

        verdicts[claim] = verify_claim(claim, evidence_list)

    elapsed = time.time() - start_time

    # Print summary
    counts = Counter(v["verdict"] for v in verdicts.values())

    print(f"✅ Verification complete in {elapsed:.1f}s")
    print(
        f"   SUPPORTED: {counts.get('SUPPORTED', 0)} | "
        f"CONTRADICTED: {counts.get('CONTRADICTED', 0)} | "
        f"UNVERIFIABLE: {counts.get('UNVERIFIABLE', 0)} | "
        f"INSUFFICIENT_EVIDENCE: {counts.get('INSUFFICIENT_EVIDENCE', 0)}"
    )

    return {"verdicts": verdicts}


def verify_claim(claim: str, evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Verify a single claim against its evidence.
    Single LLM call per claim.
    """

    # No evidence at all
    if not evidence_list:
        return {
            "claim": claim,
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "reasoning": "No Wikipedia evidence found for this claim.",
            "evidence_used": [],
        }

    # Filter out the "No evidence found" fallback
    real_evidence = [e for e in evidence_list if e.get("relevance", 0) > 0]

    if not real_evidence:
        return {
            "claim": claim,
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "reasoning": "No relevant Wikipedia evidence found.",
            "evidence_used": [],
        }

    # Get evidence quality stats
    max_relevance = max(e.get("relevance", 0) for e in real_evidence)

    # All evidence below 40% → skip LLM call, mark unverifiable
    if max_relevance < 0.40:
        return {
            "claim": claim,
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "reasoning": f"Evidence relevance too low ({max_relevance:.1%}). "
            f"Retrieved articles do not specifically address this claim.",
            "evidence_used": [e["title"] for e in real_evidence[:2]],
        }

    # One LLM call that handles both relevance check AND verification
    result = verify_with_evidence_combined(claim, real_evidence[:3])

    # Normalize verdict to handle typos (CONTRADADICTED → CONTRADICTED)
    result["verdict"] = normalize_verdict(
        result.get("verdict", "INSUFFICIENT_EVIDENCE")
    )

    # Map IRRELEVANT to UNVERIFIABLE (consistent verdict types)
    if result.get("verdict") == "IRRELEVANT":
        result["verdict"] = "UNVERIFIABLE"
        result["reasoning"] = (
            "Evidence retrieved is about a different topic: "
            + result.get("reasoning", "")
        )

    # Confidence penalty for medium-quality evidence (< 80% relevance)
    if max_relevance < 0.80:
        result["confidence"] = min(result.get("confidence", 0.5), 0.7)
        result["reasoning"] += (
            f" (Evidence relevance: {max_relevance:.1%}, confidence capped)"
        )

    result["claim"] = claim
    result["evidence_used"] = [e["title"] for e in real_evidence[:3]]

    return result


def verify_with_evidence_combined(
    claim: str, evidence_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Single LLM call that:
    1. Checks if evidence is relevant to the claim
    2. If relevant, verifies support/contradiction with STRICT rules
    3. If not relevant, returns IRRELEVANT
    """

    # Prepare evidence
    evidence_text = ""
    for i, ev in enumerate(evidence_list[:3], 1):
        evidence_text += f"\n--- Source {i}: {ev['title']} (relevance: {ev.get('relevance', 0):.1%}) ---\n"
        evidence_text += f"{ev['content'][:400]}\n"

    prompt = f"""You are a strict fact-checker. Your task is to verify the claim against the evidence.

CRITICAL RULES - FOLLOW EXACTLY:

1. FIRST: Check if evidence is about the same topic as the claim.
   - If completely different topic → verdict: "IRRELEVANT"

2. SECOND: If evidence is relevant, compare EVERY fact in the claim:

   DATE CHECKING (STRICTEST):
   - Extract ALL years/dates from claim
   - Extract ALL years/dates from evidence
   - ANY year mismatch → verdict: "CONTRADICTED"
   - Example: Claim says "2019", evidence says "2020" → CONTRADICTED
   - Example: Claim says "developed in 2019", evidence says "released in 2020" → CONTRADICTED
   - No exceptions for off-by-one-year

   NUMBER CHECKING:
   - Extract ALL numbers from claim (parameters, layers, percentages, etc.)
   - Extract ALL numbers from evidence
   - Any number mismatch > 5% → verdict: "CONTRADICTED"
   - Claim says "175 billion" but evidence says "175 million" → CONTRADICTED
   - Claim says "12 layers" but evidence says "96 layers" → CONTRADICTED

   NAME CHECKING:
   - Extract ALL named entities (people, companies, products)
   - Claim says "developed by OpenAI", evidence says "developed by Google" → CONTRADICTED
   - Any creator/originator mismatch → CONTRADICTED

   VERDICT DEFINITIONS:
   - SUPPORTED: ALL facts in claim match evidence (dates, numbers, names)
   - CONTRADICTED: ANY fact in claim contradicts evidence
   - INSUFFICIENT_EVIDENCE: Evidence is relevant but missing specific details to confirm/deny
   - IRRELEVANT: Evidence is about a different topic

CLAIM TO VERIFY:
"{claim}"

EVIDENCE:
{evidence_text}

Return JSON only:
{{
    "verdict": "SUPPORTED" or "CONTRADICTED" or "INSUFFICIENT_EVIDENCE" or "IRRELEVANT",
    "confidence": 0.0 to 1.0,
    "reasoning": "List EVERY date, number, and name in the claim. Then list what the evidence says for each. Point out any mismatches explicitly.",
    "key_facts_matched": ["specific facts from evidence that match claim"],
    "discrepancies": ["specific contradictions - include the claim value and evidence value"]
}}

Example reasoning for CONTRADICTED:
"Claim states GPT-3 was developed in 2019. Evidence states GPT-3 was released in 2020. Year mismatch: 2019 ≠ 2020."

Example reasoning for SUPPORTED:
"Claim states GPT-3 has 175 billion parameters. Evidence confirms 175 billion parameters. No date mismatch found. No name mismatch found."

Return ONLY the JSON object, no other text."""

    try:
        response = llm.invoke(prompt)
        content = clean_response(response.content)

        # Extract JSON
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))

            # Apply normalization immediately
            verdict = normalize_verdict(result.get("verdict", "INSUFFICIENT_EVIDENCE"))

            # Additional check: if LLM says SUPPORTED but has discrepancies, force CONTRADICTED
            discrepancies = result.get("discrepancies", [])
            if verdict == "SUPPORTED" and discrepancies:
                verdict = "CONTRADICTED"
                result["reasoning"] = f"Contradictions found but marked SUPPORTED. {result.get('reasoning', '')}"

            return {
                "verdict": verdict,
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", "No reasoning provided."),
                "key_facts_matched": result.get("key_facts_matched", []),
                "discrepancies": discrepancies,
            }
    except Exception as e:
        print(f"    Verification failed: {e}")

    # Fallback
    return {
        "verdict": "INSUFFICIENT_EVIDENCE",
        "confidence": 0.0,
        "reasoning": "Verification process encountered an error.",
        "key_facts_matched": [],
        "discrepancies": [],
    }


def clean_response(text: str) -> str:
    """Remove model artifacts."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text.strip()

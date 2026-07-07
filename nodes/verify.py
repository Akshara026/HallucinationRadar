"""
verify.py - Claim Verification Node

Verifies each claim against retrieved Wikipedia evidence.
Single LLM call per claim (combines relevance check + verification).
Handles evidence quality with confidence capping.
"""

import json
import re
import time
from collections import Counter
from typing import Any, Dict, List

from langchain_ollama import ChatOllama

# Deterministic model for verification
llm = ChatOllama(model="qwen2.5:7b", temperature=0, num_ctx=4096)


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
    2. If relevant, verifies support/contradiction
    3. If not relevant, returns IRRELEVANT
    """

    # Prepare evidence
    evidence_text = ""
    for i, ev in enumerate(evidence_list[:3], 1):
        evidence_text += f"\n--- Source {i}: {ev['title']} (relevance: {ev.get('relevance', 0):.1%}) ---\n"
        evidence_text += f"{ev['content'][:400]}\n"

    prompt = f"""You are a fact-checker. Your task has TWO steps:

STEP 1: Determine if the evidence is about the same topic as the claim.
- If the evidence is about something completely different (wrong person, wrong concept, wrong field), it's IRRELEVANT.
- If the evidence shares the same topic but focuses on different aspects, it's RELEVANT but may be insufficient.

STEP 2: If evidence is relevant, verify the claim.
- SUPPORTED: Evidence confirms the claim with matching facts
- CONTRADICTED: Evidence states different facts that conflict with the claim
- INSUFFICIENT_EVIDENCE: Evidence is about the right topic but doesn't have enough detail

CLAIM TO VERIFY:
"{claim}"

EVIDENCE:
{evidence_text}

Return JSON only:
{{
    "verdict": "SUPPORTED" or "CONTRADICTED" or "INSUFFICIENT_EVIDENCE" or "IRRELEVANT",
    "confidence": 0.0 to 1.0,
    "reasoning": "Explain your decision. If IRRELEVANT, explain what the evidence is actually about.",
    "key_facts_matched": ["fact1 from evidence that matches claim"],
    "discrepancies": ["any contradictions between claim and evidence"]
}}

Rules:
- IRRELEVANT means the evidence is about a different topic entirely
- Check dates and numbers precisely (±1 year for dates)
- Be strict: partial support with wrong details = CONTRADICTED
- Return ONLY the JSON object, no other text"""

    try:
        response = llm.invoke(prompt)
        content = clean_response(response.content)

        # Extract JSON
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            return {
                "verdict": result.get("verdict", "INSUFFICIENT_EVIDENCE"),
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", "No reasoning provided."),
                "key_facts_matched": result.get("key_facts_matched", []),
                "discrepancies": result.get("discrepancies", []),
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

# this is actual code. but have been modified in verfy.py coz of hardware issue Y_Y . here we calling 2 llms. but my systemn cant handle 2 llm call

"""
verify.py - Claim Verification Node

Verifies each claim against retrieved Wikipedia evidence.
Handles three evidence quality scenarios:
1. High relevance (>75%) → Actually check support/contradiction
2. Medium relevance (40-75%) → LLM checks if evidence is even relevant
3. Low relevance (<40%) → Mark as unverifiable

Outputs nuanced verdicts: SUPPORTED, CONTRADICTED, UNVERIFIABLE, INSUFFICIENT_EVIDENCE
"""

import json
import re
import time
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
    supported = sum(1 for v in verdicts.values() if v["verdict"] == "SUPPORTED")
    contradicted = sum(1 for v in verdicts.values() if v["verdict"] == "CONTRADICTED")
    unverifiable = sum(1 for v in verdicts.values() if v["verdict"] == "UNVERIFIABLE")
    insufficient = sum(
        1 for v in verdicts.values() if v["verdict"] == "INSUFFICIENT_EVIDENCE"
    )

    print(f"✅ Verification complete in {elapsed:.1f}s")
    print(
        f"   Supported: {supported} | Contradicted: {contradicted} | "
        f"Unverifiable: {unverifiable} | Insufficient: {insufficient}"
    )

    return {"verdicts": verdicts}


def verify_claim(claim: str, evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Verify a single claim against its evidence.

    Decision logic:
    - No evidence → UNVERIFIABLE
    - All evidence low relevance (<40%) → UNVERIFIABLE
    - Medium relevance → Check if evidence is actually about the claim
    - High relevance → Full verification against evidence content
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

    # Check evidence quality
    max_relevance = max(e.get("relevance", 0) for e in real_evidence)
    high_quality = [e for e in real_evidence if e.get("relevance", 0) >= 0.75]
    medium_quality = [e for e in real_evidence if 0.40 <= e.get("relevance", 0) < 0.75]

    # Case 1: High quality evidence → Full verification
    if high_quality:
        return verify_with_evidence(claim, high_quality)

    # Case 2: Medium quality → Check if evidence is actually relevant
    if medium_quality:
        relevance_check = check_evidence_relevance(claim, medium_quality)

        if relevance_check["is_relevant"]:
            # Evidence is relevant, proceed with verification
            return verify_with_evidence(claim, medium_quality)
        else:
            return {
                "claim": claim,
                "verdict": "UNVERIFIABLE",
                "confidence": 0.0,
                "reasoning": relevance_check["reasoning"],
                "evidence_used": [e["title"] for e in medium_quality],
            }

    # Case 3: Low relevance → Unverifiable
    return {
        "claim": claim,
        "verdict": "UNVERIFIABLE",
        "confidence": 0.0,
        "reasoning": f"Evidence relevance too low ({max_relevance:.1%}). "
        f"Retrieved articles do not specifically address this claim.",
        "evidence_used": [e["title"] for e in real_evidence[:2]],
    }


def check_evidence_relevance(
    claim: str, evidence_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Check if medium-relevance evidence is actually about the claim.
    Prevents trusting 77% relevance on wrong articles (like NMIMS for LLM claims).
    """

    # Prepare evidence content
    evidence_text = ""
    for i, ev in enumerate(evidence_list[:2], 1):
        evidence_text += (
            f"Evidence {i} (from '{ev['title']}'):\n{ev['content'][:400]}\n\n"
        )

    prompt = f"""Determine if the following evidence is actually relevant to verifying this claim.

Claim: "{claim}"

{evidence_text}

Is this evidence about the same topic as the claim? Or is it about something completely different
that just happens to share some words?

Return JSON only:
{{
    "is_relevant": true/false,
    "reasoning": "Brief explanation of why evidence is or isn't relevant"
}}"""

    try:
        response = llm.invoke(prompt)
        content = clean_response(response.content)

        # Extract JSON
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            return {
                "is_relevant": result.get("is_relevant", False),
                "reasoning": result.get("reasoning", "Could not determine relevance."),
            }
    except Exception as e:
        print(f"    Relevance check failed: {e}")

    # Fallback: assume not relevant
    return {
        "is_relevant": False,
        "reasoning": "Could not verify if evidence is relevant to this claim.",
    }


def verify_with_evidence(
    claim: str, evidence_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Full verification: compare claim against evidence content.
    Uses LLM to determine if evidence supports, contradicts, or is insufficient.
    """

    # Prepare evidence for the LLM
    evidence_text = ""
    sources = []

    for i, ev in enumerate(evidence_list[:3], 1):  # Max 3 evidence pieces
        sources.append(ev["title"])
        evidence_text += f"\n--- Source {i}: {ev['title']} ---\n"
        evidence_text += f"{ev['content'][:500]}\n"

    prompt = f"""You are a fact-checker. Compare this claim against the provided Wikipedia evidence.

CLAIM TO VERIFY:
"{claim}"

WIKIPEDIA EVIDENCE:
{evidence_text}

Determine if the evidence:
- SUPPORTED: Evidence confirms the claim with specific matching facts
- CONTRADICTED: Evidence states something different that contradicts the claim
- INSUFFICIENT_EVIDENCE: Evidence is related but doesn't have enough detail to confirm or contradict
- IRRELEVANT: Evidence is about a different topic entirely

Verification rules:
- Check dates, numbers, names, and specific facts precisely
- Allow minor variations (±1 year for dates, minor name variations)
- If the claim says "X is Y" and evidence says "X is Z", that's CONTRADICTED
- If evidence only discusses X but never mentions Y, that's INSUFFICIENT_EVIDENCE
- Be strict: partial support with key details wrong = CONTRADICTED

Return JSON only:
{{
    "verdict": "SUPPORTED" or "CONTRADICTED" or "INSUFFICIENT_EVIDENCE" or "IRRELEVANT",
    "confidence": 0.0 to 1.0,
    "reasoning": "Explain exactly what the evidence says and how it relates to the claim",
    "key_facts_matched": ["fact1 matched", "fact2 matched"],
    "discrepancies": ["any contradictions found or empty list"]
}}

IMPORTANT: Return ONLY the JSON object, no other text."""

    try:
        response = llm.invoke(prompt)
        content = clean_response(response.content)

        # Extract JSON
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            return {
                "claim": claim,
                "verdict": result.get("verdict", "INSUFFICIENT_EVIDENCE"),
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", "No reasoning provided."),
                "evidence_used": sources,
                "key_facts_matched": result.get("key_facts_matched", []),
                "discrepancies": result.get("discrepancies", []),
            }
    except Exception as e:
        print(f"    Verification LLM call failed: {e}")

    # Fallback
    return {
        "claim": claim,
        "verdict": "INSUFFICIENT_EVIDENCE",
        "confidence": 0.0,
        "reasoning": "Verification process failed.",
        "evidence_used": sources,
        "key_facts_matched": [],
        "discrepancies": [],
    }


def clean_response(text: str) -> str:
    """Remove any model artifacts."""
    # Remove think tags (for deepseek models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove markdown code fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text.strip()


# Quick test
if __name__ == "__main__":
    print("=" * 70)
    print("TESTING VERIFICATION NODE")
    print("=" * 70)

    # Simulate evidence from retrieval node
    test_state = {
        "claims": [
            "The Eiffel Tower was completed in 1889",
            "Python was created by Guido van Rossum in 1991",
            "LLMs handle specialized tasks efficiently",
        ],
        "evidence": {
            "The Eiffel Tower was completed in 1889": [
                {
                    "title": "Eiffel Tower",
                    "url": "https://en.wikipedia.org/wiki/Eiffel_Tower",
                    "content": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.",
                    "relevance": 0.89,
                    "source_type": "summary",
                }
            ],
            "Python was created by Guido van Rossum in 1991": [
                {
                    "title": "Python (programming language)",
                    "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                    "content": "Python was created by Guido van Rossum and first released on February 20, 1991. It is a high-level, general-purpose programming language.",
                    "relevance": 0.92,
                    "source_type": "summary",
                }
            ],
            "LLMs handle specialized tasks efficiently": [
                {
                    "title": "SVKM's NMIMS",
                    "url": "https://en.wikipedia.org/wiki/SVKM%27s_NMIMS",
                    "content": "SVKM's NMIMS is a private university located in Mumbai, India. It was established in 1981.",
                    "relevance": 0.77,
                    "source_type": "paragraph",
                }
            ],
        },
    }

    result = verify_node(test_state)

    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)

    for claim, verdict in result["verdicts"].items():
        print(f"\n📌 Claim: {claim}")
        print(f"   Verdict: {verdict['verdict']}")
        print(f"   Confidence: {verdict['confidence']:.1%}")
        print(f"   Reasoning: {verdict['reasoning'][:150]}...")
        if verdict.get("evidence_used"):
            print(f"   Sources: {', '.join(verdict['evidence_used'])}")

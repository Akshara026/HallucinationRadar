"""
verify.py - Claim Verification Node

Verifies each claim against retrieved Wikipedia evidence.
Single LLM call per claim (combines relevance check + verification).
Handles evidence quality with confidence capping.
Strict date/number matching to prevent false SUPPORTED verdicts.
Enforces symmetric consistency between verdict and discrepancies.
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

    if "CONTRADICT" in verdict or "CONTRADADICT" in verdict:
        return "CONTRADICTED"
    if "SUPPORT" in verdict:
        return "SUPPORTED"
    if "UNVERIF" in verdict:
        return "UNVERIFIABLE"
    if "INSUFFICIENT" in verdict or "INSUFFICENT" in verdict:
        return "INSUFFICIENT_EVIDENCE"
    if "IRRELEVANT" in verdict:
        return "UNVERIFIABLE"

    if verdict not in VALID_VERDICTS:
        return "INSUFFICIENT_EVIDENCE"

    return verdict


def clamp_confidence(value: Any) -> float:
    """Ensure confidence is always a valid float between 0.0 and 1.0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, v))


def verify_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify all claims against their retrieved evidence.

    Input:  state["claims"] - list of claims
            state["evidence"] - dict mapping claim -> list of evidence dicts

    Output: state["verdicts"] - dict mapping claim -> verification result
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

    if not evidence_list:
        return {
            "claim": claim,
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "reasoning": "No Wikipedia evidence found for this claim.",
            "evidence_used": [],
            "key_facts_matched": [],
            "discrepancies": [],
        }

    # arXiv abstracts are dense and technical — they can superficially keyword-match
    # without being topically relevant, more so than Wikipedia intro text. Require a
    # higher bar for arXiv evidence to be considered "real" evidence at all.
    real_evidence = []
    for e in evidence_list:
        relevance = e.get("relevance", 0)
        source = e.get("source", "wikipedia")
        min_bar = 0.55 if source == "arxiv" else 0.0
        if relevance > min_bar:
            real_evidence.append(e)

    if not real_evidence:
        return {
            "claim": claim,
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "reasoning": "No relevant evidence found.",
            "evidence_used": [],
            "key_facts_matched": [],
            "discrepancies": [],
        }

    max_relevance = max(e.get("relevance", 0) for e in real_evidence)

    # All evidence below 40% -> skip LLM call entirely
    if max_relevance < 0.40:
        return {
            "claim": claim,
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "reasoning": f"Evidence relevance too low ({max_relevance:.1%}). "
            f"Retrieved sources do not specifically address this claim.",
            "evidence_used": [e["title"] for e in real_evidence[:2]],
            "key_facts_matched": [],
            "discrepancies": [],
        }

    result = verify_with_evidence_combined(claim, real_evidence[:3])

    result["verdict"] = normalize_verdict(
        result.get("verdict", "INSUFFICIENT_EVIDENCE")
    )

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

    result["confidence"] = clamp_confidence(result.get("confidence"))
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

    Enforces symmetric consistency:
    - SUPPORTED + real discrepancies -> forced to CONTRADICTED
    - CONTRADICTED + no discrepancies -> forced to INSUFFICIENT_EVIDENCE
    """

    evidence_text = ""
    for i, ev in enumerate(evidence_list[:3], 1):
        evidence_text += f"\n--- Source {i}: {ev['title']} (relevance: {ev.get('relevance', 0):.1%}) ---\n"
        evidence_text += f"{ev['content'][:400]}\n"

    prompt = f"""You are a strict fact-checker. Your task is to verify the claim against the evidence.

CRITICAL RULES - FOLLOW EXACTLY:

1. FIRST: Check if evidence is about the same topic as the claim.
   - If completely different topic -> verdict: "IRRELEVANT"

2. SECOND: If evidence is relevant, verify ONLY the specific facts stated in the claim.

CONTRADICTION RULES (STRICT):
- ONLY mark CONTRADICTED if the claim states a SPECIFIC fact (date, number, name)
  that DIRECTLY conflicts with a SPECIFIC fact in evidence
- Do NOT mark CONTRADICTED just because evidence mentions additional details not in the claim
- If evidence is SILENT on something the claim states -> INSUFFICIENT_EVIDENCE
- If evidence mentions EXTRA facts not in the claim -> do NOT mark CONTRADICTED
- Before marking CONTRADICTED, you MUST state exactly which specific fact conflicts
  in the "discrepancies" field
- If you cannot state a specific conflicting fact -> use INSUFFICIENT_EVIDENCE instead

DATE CHECKING:
- Extract years/dates FROM THE CLAIM
- Only check dates THAT THE CLAIM EXPLICITLY STATES
- If claim says "2019" and evidence says "2020" -> CONTRADICTED
- If claim does NOT mention a date, do NOT invent a date mismatch

NUMBER CHECKING:
- Extract numbers FROM THE CLAIM
- Only check numbers THAT THE CLAIM EXPLICITLY STATES
- If claim says "17 billion" and evidence says "175 billion" -> CONTRADICTED
- If claim does NOT mention a number, do NOT invent a number mismatch

NAME CHECKING:
- Extract named entities FROM THE CLAIM
- Only check names THAT THE CLAIM EXPLICITLY STATES
- If claim says "OpenAI" and evidence says "Google" -> CONTRADICTED
- If claim does NOT mention a creator, do NOT invent a name mismatch

VERDICT DEFINITIONS:
- SUPPORTED: Evidence EXPLICITLY confirms the specific facts stated in the claim.
  Being about the same general topic is NOT enough — the evidence must actually
  state the specific detail the claim makes.
- CONTRADICTED: At least ONE specific fact in claim DIRECTLY conflicts with evidence
- INSUFFICIENT_EVIDENCE: Evidence is relevant/related but does NOT explicitly confirm
  or deny the specific facts in the claim. This is the DEFAULT when evidence is
  merely topically related without confirming the actual detail.
- IRRELEVANT: Evidence is about a different topic entirely

SELF-CHECK BEFORE RETURNING SUPPORTED:
- Does the evidence EXPLICITLY state the specific fact(s) the claim makes, not just
  discuss the same general topic?
- If your own reasoning includes phrases like "does not directly address", "does not
  explicitly state", or "does not mention" — that is INSUFFICIENT_EVIDENCE, not SUPPORTED.
- When in doubt between SUPPORTED and INSUFFICIENT_EVIDENCE, choose INSUFFICIENT_EVIDENCE.

SELF-CHECK BEFORE RETURNING CONTRADICTED:
- Does the claim make a specific factual assertion that evidence EXPLICITLY disagrees with?
- Can I state the exact contradiction in the "discrepancies" field?
- If NO to either question -> do NOT use CONTRADICTED, use INSUFFICIENT_EVIDENCE instead

CLAIM TO VERIFY:
"{claim}"

EVIDENCE:
{evidence_text}

Return JSON only:
{{
    "verdict": "SUPPORTED" or "CONTRADICTED" or "INSUFFICIENT_EVIDENCE" or "IRRELEVANT",
    "confidence": 0.0 to 1.0,
    "reasoning": "State what the claim says. Then state what the evidence says. Only point out mismatches for facts the claim EXPLICITLY states.",
    "key_facts_matched": ["specific facts from evidence that match the claim"],
    "discrepancies": ["ONLY list contradictions where the claim states a specific fact and evidence DIRECTLY disagrees"]
}}

Return ONLY the JSON object, no other text."""

    try:
        response = llm.invoke(prompt)
        content = clean_response(response.content)

        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))

            verdict = normalize_verdict(result.get("verdict", "INSUFFICIENT_EVIDENCE"))
            discrepancies = result.get("discrepancies", []) or []
            reasoning = result.get("reasoning", "No reasoning provided.")

            # Symmetric enforcement between verdict and discrepancies:

            # Case 1: SUPPORTED but real discrepancies exist -> should be CONTRADICTED
            if verdict == "SUPPORTED" and discrepancies:
                verdict = "CONTRADICTED"
                reasoning = f"Contradictions found but model marked SUPPORTED (auto-corrected). {reasoning}"

            # Case 2: CONTRADICTED but no discrepancies stated -> model broke its own rule,
            # downgrade to INSUFFICIENT_EVIDENCE per the self-check instruction
            elif verdict == "CONTRADICTED" and not discrepancies:
                verdict = "INSUFFICIENT_EVIDENCE"
                reasoning = f"Marked CONTRADICTED without a stated discrepancy (auto-corrected to INSUFFICIENT_EVIDENCE). {reasoning}"

            # Case 3: SUPPORTED but the model's own reasoning admits evidence doesn't
            # actually confirm the detail (hedging language) -> downgrade to INSUFFICIENT_EVIDENCE.
            # This is a code-level backstop in case the LLM ignores its self-check instruction.
            hedge_patterns = [
                r"does not (directly )?(explicitly )?(address|mention|state|provide|confirm|specify)",
                r"none of the (provided )?(evidence|sources?) (directly |explicitly )?(mention|address|state|confirm|discuss)",
                r"no (specific |direct |explicit )?(evidence|mention|indication) (that|of|explicitly)",
                r"not (explicitly |directly )?(stated|mentioned|specified|confirmed|addressed)",
                r"(evidence|source)s? (do|does) not (directly |explicitly )?(mention|address|state|confirm)",
                r"no (provided )?(evidence|source) (directly |explicitly )?(mentions?|addresses?|states?|confirms?)",
                r"without (explicitly |directly )?(mentioning|stating|confirming)",
            ]
            reasoning_lower = reasoning.lower()
            hedged = any(re.search(p, reasoning_lower) for p in hedge_patterns)

            if verdict == "SUPPORTED" and hedged:
                verdict = "INSUFFICIENT_EVIDENCE"
                reasoning = f"Auto-corrected: marked SUPPORTED but reasoning admits evidence doesn't confirm the specific detail. {reasoning}"

            return {
                "verdict": verdict,
                "confidence": clamp_confidence(result.get("confidence", 0.5)),
                "reasoning": reasoning,
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

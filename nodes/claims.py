"""
claims.py - Claim Extraction Node

Extracts atomic factual claims from generated answer text.
Filters out vague/unverifiable claims while keeping verifiable ones.
Splits compound claims into individual verifiable facts, including
statistics embedded as parentheticals or mid-sentence asides.
"""

import re
from typing import Any, Dict, List

from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:7b", temperature=0)


def claims_node(state: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract atomic factual claims from the answer text.
    Each claim should be independently verifiable.
    Filters out vague/unverifiable claims.
    Splits compound claims.
    """
    answer = state.get("answer", "")

    if not answer or not answer.strip():
        return {"claims": []}

    answer = answer.strip()

    prompt = f"""Extract atomic, verifiable factual claims from the following text.

CRITICAL EXTRACTION RULES:
1. Each claim must contain EXACTLY ONE verifiable fact
2. Break compound sentences into separate claims
3. SPLIT claims connected by "while", "and", "or", "but" when they contain TWO separate facts
4. Remove opinions, speculations, and subjective statements
5. Each claim must be self-contained (no pronouns without clear referents)
6. Replace pronouns (he, she, it, they, this, that) with their specific referents
7. Preserve exact numbers, dates, and proper nouns from the source
8. Include necessary context so each claim can be verified independently
9. Format: Return ONLY the claims, one per line, no numbering or bullet points
10. SKIP vague claims that cannot be fact-checked (see examples below)
11. EXTRACT every specific number or statistic as its OWN separate claim, even if it
    only appears as a parenthetical aside, an example, or is buried mid-sentence.
    Restate the subject explicitly so the claim stands alone. Do NOT leave a specific
    number embedded inside a larger claim about something else — pull it out.

EMBEDDED STATISTIC EXTRACTION EXAMPLES (this is a common mistake — watch for it):
Input: "These models process vast amounts of text, such as 3 billion tokens from books and Wikipedia"
Output:
These models process vast amounts of text from books and Wikipedia.
These models are trained on 3 billion tokens.

Input: "These models often consist of multiple layers, such as 12 in GPT-4, each handling specific functions"
Output:
These models often consist of multiple layers, each handling specific functions.
GPT-4 has 12 layers.

Input: "The model achieved strong results (92% accuracy) on the benchmark"
Output:
The model achieved strong results on the benchmark.
The model achieved 92% accuracy on the benchmark.

COMPOUND CLAIM SPLITTING EXAMPLES:
Input: "The smallest version has 7 billion parameters, while the largest has over 100 billion"
Output:
The smallest version has 7 billion parameters.
The largest version has over 100 billion parameters.

Input: "GPT-3 was developed by OpenAI in 2019 and has 175 billion parameters"
Output:
GPT-3 was developed by OpenAI in 2019.
GPT-3 has 175 billion parameters.

Input: "LLMs use transformers for processing and are trained on large datasets"
Output:
LLMs use transformers for processing.
LLMs are trained on large datasets.

EXAMPLES OF CLAIMS TO SKIP (too vague to verify):
- "The evolution of LLMs highlights the growing potential of AI"
- "Advancements have enabled LLMs to scale effectively"
- "LLMs represent a significant breakthrough in AI"
- "The future of LLMs looks promising"
- "LLMs have transformed the field of NLP"

EXAMPLES OF GOOD CLAIMS (specific and verifiable):
- "GPT-3 was developed by OpenAI"
- "BERT was released by Google in 2018"
- "The transformer architecture was introduced in 2017"
- "GPT-3 has 175 billion parameters"
- "LLMs are trained on large text corpora including books and Wikipedia"

TEXT TO PROCESS:
{answer}

RETURN ONLY THE EXTRACTED CLAIMS (one per line):"""

    try:
        response = llm.invoke(prompt)

        claims = extract_claims_from_response(response.content)

        # Split compound claims (while/and with numbers on both sides)
        claims = split_compound_claims(claims)

        # Code-level safety net: split out any remaining embedded statistics
        # the LLM might have missed (parentheticals, "such as N X" asides)
        claims = split_embedded_statistics(claims)

        # Post-process claims
        claims = clean_and_validate_claims(claims)

        # Filter out vague/unverifiable claims
        claims = filter_vague_claims(claims)

        print(f"  Extracted {len(claims)} verifiable claims")

        return {"claims": claims}

    except Exception as e:
        print(f"Error extracting claims: {e}")
        return {"claims": []}


def split_compound_claims(claims: List[str]) -> List[str]:
    """
    Split compound claims into atomic claims.
    Detects patterns like "X while Y", "X and Y" where both are separate facts.
    """
    split_claims = []

    for claim in claims:
        # Split on "while" when both sides have numbers/facts
        if " while " in claim.lower():
            parts = re.split(r'\s+while\s+', claim, flags=re.IGNORECASE)
            if len(parts) == 2:
                if (re.search(r'\d+', parts[0]) and re.search(r'\d+', parts[1])):
                    split_claims.append(parts[0].strip().rstrip('.') + ".")
                    split_claims.append(parts[1].strip().rstrip('.') + ".")
                    continue

        # Split on " and " when both sides have numbers
        if " and " in claim.lower():
            parts = re.split(r'\s+and\s+', claim, flags=re.IGNORECASE)
            if len(parts) == 2:
                if (re.search(r'\d+', parts[0]) and re.search(r'\d+', parts[1])):
                    if len(parts[0].split()) > 3 and len(parts[1].split()) > 3:
                        split_claims.append(parts[0].strip().rstrip('.') + ".")
                        split_claims.append(parts[1].strip().rstrip('.') + ".")
                        continue

        split_claims.append(claim)

    return split_claims


# Patterns that mark a number as a parenthetical/aside rather than the
# main point of the sentence — these are exactly the cases that slip
# through as unverified hallucinated statistics if not pulled out.
_EMBEDDED_STAT_PATTERNS = [
    # "..., such as 12 in GPT-4, ..." / "..., such as 3 billion tokens ..."
    re.compile(
        r"^(?P<subject>.*?),?\s+such as\s+(?P<stat>[\d,\.]+\s*(?:billion|million|thousand|trillion)?\s*[a-zA-Z][\w\s]*?)(?:,|\.|$)(?P<rest>.*)$",
        re.IGNORECASE,
    ),
    # "... (92% accuracy) ..." style parenthetical stat
    re.compile(
        r"^(?P<subject>.*?)\((?P<stat>[\d,\.]+%?\s*[a-zA-Z][\w\s]*?)\)(?P<rest>.*)$"
    ),
]


def split_embedded_statistics(claims: List[str]) -> List[str]:
    """
    Safety net: catch specific numbers/statistics still embedded as an
    aside inside a larger claim (parentheses, "such as N X") and pull
    them out into their own standalone claim. This is a backstop in
    case the LLM's own extraction misses one of these patterns.
    """
    result = []

    for claim in claims:
        text = claim.strip()
        matched = False

        for pattern in _EMBEDDED_STAT_PATTERNS:
            m = pattern.match(text)
            if not m:
                continue

            subject = m.group("subject").strip().rstrip(",").strip()
            stat = m.group("stat").strip()
            rest = m.group("rest").strip().lstrip(",").strip()

            # Need a real number in the extracted stat, and a real subject
            # left over, or this isn't a genuine embedded-statistic case
            if not re.search(r"\d", stat) or len(subject.split()) < 3:
                continue

            base_sentence = (subject + (" " + rest if rest else "")).strip()
            base_sentence = re.sub(r"\s+", " ", base_sentence).strip().rstrip(".") + "."
            stat_sentence = f"{subject.split(',')[0].strip()} involves {stat}.".strip()

            # Only split if both halves are substantive — otherwise keep original
            if len(base_sentence.split()) >= 4:
                result.append(base_sentence)
                result.append(stat_sentence[0].upper() + stat_sentence[1:])
                matched = True
                break

        if not matched:
            result.append(claim)

    return result


def extract_claims_from_response(text: str) -> List[str]:
    """Extract and clean individual claims from the LLM response."""
    claims = []

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        line = re.sub(r"^[\d]+[\.\)]\s*", "", line)  # Remove numbering
        line = re.sub(r"^[•\-\*\✓\✅\❌\⭐\►]\s*", "", line)  # Remove bullet points
        line = re.sub(r"^[Ff]act:\s*", "", line)  # Remove "Fact:" prefix
        line = re.sub(r"^[Cc]laim:\s*", "", line)  # Remove "Claim:" prefix
        line = line.strip('"\'""')  # Remove quotes

        if line and len(line.split()) >= 3:
            claims.append(line)

    return claims


def clean_and_validate_claims(claims: List[str]) -> List[str]:
    """Post-process claims to ensure quality and deduplication."""
    cleaned = []
    seen = set()

    for claim in claims:
        if not claim.endswith((".", "!", "?")):
            claim += "."

        claim = claim[0].upper() + claim[1:] if claim else claim

        normalized = claim.lower().strip()
        if normalized not in seen:
            seen.add(normalized)

            if is_likely_factual(claim):
                cleaned.append(claim)

    return cleaned


def is_likely_factual(claim: str) -> bool:
    """
    Check if a claim appears to be factual rather than opinion.
    Uses multiple signals to determine if claim is verifiable.
    """
    opinion_indicators = [
        "i think",
        "i believe",
        "in my opinion",
        "arguably",
        "perhaps",
        "maybe",
        "probably",
        "could be",
        "might be",
        "seems to",
        "appears to",
    ]

    claim_lower = claim.lower()
    for indicator in opinion_indicators:
        if indicator in claim_lower:
            return False

    has_number = bool(re.search(r"\d+", claim))
    has_proper_noun = bool(re.search(r"\b[A-Z][a-zA-Z]+\b", claim))
    is_long_enough = len(claim.split()) >= 8
    has_technical_term = bool(
        re.search(
            r"\b(transformer|neural|attention|token|parameter|layer|dataset|training|model|architecture)\b",
            claim_lower,
        )
    )

    return has_number or has_proper_noun or is_long_enough or has_technical_term


def filter_vague_claims(claims: List[str]) -> List[str]:
    """
    Filter out claims that are too vague to verify.
    Uses pattern matching to catch common vague claim structures.
    Does NOT double-filter claims that already passed is_likely_factual.
    """
    vague_patterns = [
        r"^the (evolution|development|advancement|progress|future|rise|growth) of",
        r"^advancements? (have|has) enabled",
        r"^LLMs (represent|are|have become)",
        r"(highlights?|demonstrates?|shows?|illustrates?) the (growing|increasing|potential|importance)",
        r"^the (field|area|domain) of",
        r"(transformed|revolutionized|changed) the (field|way|landscape)",
        r"^as (AI|technology|LLMs) (continues|continue) to",
        r"^ongoing (research|development|advancements)",
        r"^future (developments|directions|work|research)",
        r"^despite (these|the) (advancements|progress|challenges)",
    ]

    filtered = []
    for claim in claims:
        claim_lower = claim.lower()
        is_vague = False

        for pattern in vague_patterns:
            if re.match(pattern, claim_lower):
                is_vague = True
                break

        if not is_vague:
            filtered.append(claim)

    return filtered

import re
from typing import Any, Dict, List

from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:7b", temperature=0)


def claims_node(state: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract atomic factual claims from the answer text.
    Each claim should be independently verifiable.
    Filters out vague/unverifiable claims.
    """
    answer = state.get("answer", "")

    if not answer or not answer.strip():
        return {"claims": []}

    answer = answer.strip()

    prompt = f"""Extract atomic, verifiable factual claims from the following text.

CRITICAL EXTRACTION RULES:
1. Each claim must contain EXACTLY ONE verifiable fact
2. Break compound sentences into separate claims
3. Remove opinions, speculations, and subjective statements
4. Each claim must be self-contained (no pronouns without clear referents)
5. Replace pronouns (he, she, it, they, this, that) with their specific referents
6. Preserve exact numbers, dates, and proper nouns from the source
7. Include necessary context so each claim can be verified independently
8. Format: Return ONLY the claims, one per line, no numbering or bullet points
9. SKIP vague claims that cannot be fact-checked (see examples below)

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

        # Post-process claims
        claims = clean_and_validate_claims(claims)

        # Filter out vague/unverifiable claims
        claims = filter_vague_claims(claims)

        print(f"  Extracted {len(claims)} verifiable claims")

        return {"claims": claims}

    except Exception as e:
        print(f"Error extracting claims: {e}")
        return {"claims": []}


def extract_claims_from_response(text: str) -> List[str]:
    """Extract and clean individual claims from the LLM response."""
    claims = []

    for line in text.split("\n"):
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        line = re.sub(r"^[\d]+[\.\)]\s*", "", line)  # Remove numbering
        line = re.sub(r"^[•\-\*\✓\✅\❌\⭐\►]\s*", "", line)  # Remove bullet points
        line = line.strip('"\'""')  # Remove quotes

        if line and len(line.split()) >= 3:  # Minimum 3 words for a valid claim
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

        # Check for duplicates
        normalized = claim.lower().strip()
        if normalized not in seen:
            seen.add(normalized)

            # Validate claim is factual (contains at least one factual indicator)
            if is_likely_factual(claim):
                cleaned.append(claim)

    return cleaned


def is_likely_factual(claim: str) -> bool:
    """Check if a claim appears to be factual rather than opinion."""
    # Skip claims that are clearly opinions or subjective
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
        "best",
        "worst",
        "greatest",
        "may struggle",
        "might lose",
        "may have errors",
        "may lack",
    ]

    claim_lower = claim.lower()

    # Check for opinion indicators
    for indicator in opinion_indicators:
        if indicator in claim_lower:
            return False

    # Must contain at least one factual element (number, proper noun, or specific entity)
    has_number = bool(re.search(r"\d+", claim))
    has_proper_noun = bool(re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", claim))

    return has_number or has_proper_noun or len(claim.split()) > 5


def filter_vague_claims(claims: List[str]) -> List[str]:
    """
    Filter out claims that are too vague to verify.
    Uses pattern matching to catch common vague claim structures.
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

        # Also check if claim has no specific entities
        if not is_vague:
            # Must have at least one of: number, proper noun, or specific technical term
            has_specifics = (
                bool(re.search(r"\d+", claim))  # Number
                or bool(re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", claim))  # Proper noun
                or bool(re.search(r"\b[A-Z]{2,}\b", claim))  # Acronym (GPT, BERT, LLM)
            )
            if has_specifics:
                filtered.append(claim)

    return filtered

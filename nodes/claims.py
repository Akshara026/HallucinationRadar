import re
from typing import Any, Dict, List

from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:7b", temperature=0)


def claims_node(state: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract atomic factual claims from the answer text.
    Each claim should be independently verifiable.
    """
    answer = state.get("answer", "")

    if not answer or not answer.strip():
        return {"claims": []}

    answer = answer.strip()  # just cleanin the ans

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

EXAMPLES:
Bad: "The company was founded in 1995 and grew rapidly"
Good:
The company was founded in 1995
The company experienced rapid growth after founding

Bad: "He invented the telephone in 1876"
Good:
Alexander Graham Bell invented the telephone in 1876

Bad: "It was the largest earthquake ever recorded"
Good:
The 1960 Valdivia earthquake was the largest earthquake ever recorded

Bad: "The research shows promising results"
Good:
[No objective claim - skip subjective statements]

TEXT TO PROCESS:
{answer}

RETURN ONLY THE EXTRACTED CLAIMS (one per line):"""

    try:
        response = llm.invoke(prompt)

        claims = extract_claims_from_response(response.content)  # Extractin claim

        # Post-process claims
        claims = clean_and_validate_claims(claims)

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

        line = re.sub(r"^[\d]+[\.\)]\s*", "", line)  # removin numbering
        line = re.sub(r"^[•\-\*\✓\✅\❌\⭐\►]\s*", "", line)  # Removin bullet points
        line = line.strip('"\'""')  # Removin quotes

        if (
            line and len(line.split()) >= 3
        ):  # thr should be min 3 words for a valid claim
            claims.append(line)

    return claims


def clean_and_validate_claims(claims: List[str]) -> List[str]:
    """Post-process claims to ensure quality and deduplication."""
    cleaned = []
    seen = set()

    for claim in claims:
        if not claim.endswith((".", "!", "?")):
            claim += "."  # makin sure tht claim ends with proper pulstop

        claim = (
            claim[0].upper() + claim[1:] if claim else claim
        )  # this just make first letter cap

        # checkin for duplicates
        normalized = claim.lower().strip()
        if normalized not in seen:
            seen.add(normalized)

            # Validate claim is factual (contains at least one factual indicator)
            if is_likely_factual(claim):
                cleaned.append(claim)

    return cleaned


def is_likely_factual(claim: str) -> bool:
    """Check if a claim appears to be factual rather than opinion."""
    # Skip claims that are clearly opinions or subjective based on below thing
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

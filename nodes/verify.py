import json

from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama3.1", temperature=0)


def verify_node(state):
    claims = state["claims"]
    evidence = state["evidence"]
    verdicts = {}

    for claim in claims:
        snippets = evidence.get(claim, [])

        if not snippets or snippets == ["No evidence found."]:
            verdicts[claim] = {
                "verdict": "unverifiable",
                "confidence": 0.0,
                "reason": "No evidence could be retrieved for this claim.",
            }
            continue

        # Build evidence block
        evidence_block = "\n\n".join(
            f"[Source {i + 1}]: {s}" for i, s in enumerate(snippets)
        )

        prompt = f"""
You are a fact-checking engine. Your job is to verify a claim against evidence.

Claim:
{claim}

Evidence:
{evidence_block}

Instructions:
- Read the evidence carefully
- Decide if the evidence supports, contradicts, or cannot verify the claim
- Give a confidence score between 0.0 and 1.0
- Give a short reason (1-2 sentences max)
- Do NOT use outside knowledge — only use the evidence provided
- Do NOT guess

Respond ONLY in this exact JSON format, nothing else:
{{
  "verdict": "supported" | "contradicted" | "unverifiable",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<short reason>"
}}
"""
        response = llm.invoke(prompt)
        raw = response.content.strip()

        try:
            # Strip markdown fences if model adds them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw.strip())

            # Validate keys exist
            verdicts[claim] = {
                "verdict": result.get("verdict", "unverifiable"),
                "confidence": float(result.get("confidence", 0.0)),
                "reason": result.get("reason", "No reason given."),
            }

        except (json.JSONDecodeError, ValueError):
            # If model fails to give valid JSON, mark it unverifiable
            verdicts[claim] = {
                "verdict": "unverifiable",
                "confidence": 0.0,
                "reason": f"Model gave unparseable response: {raw[:100]}",
            }

    return {"verdicts": verdicts}

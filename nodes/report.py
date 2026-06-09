from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama3.1", temperature=0)


def report_node(state):
    query = state["query"]
    answer = state["answer"]
    claims = state["claims"]
    verdicts = state["verdicts"]
    score = state["score"]

    # Build verdict breakdown block
    verdict_lines = []
    for claim, data in verdicts.items():
        emoji = {"supported": "✅", "contradicted": "❌", "unverifiable": "⚠️"}.get(
            data["verdict"], "❓"
        )

        line = (
            f"{emoji} [{data['verdict'].upper()}] "
            f"(confidence: {data['confidence']})\n"
            f"   Claim  : {claim}\n"
            f"   Reason : {data['reason']}"
        )
        verdict_lines.append(line)

    verdict_block = "\n\n".join(verdict_lines)

    # Score label
    if score >= 0.85:
        label = "Highly Reliable"
    elif score >= 0.60:
        label = "Mostly Reliable"
    elif score >= 0.40:
        label = "Questionable"
    else:
        label = "High Hallucination Risk"

    # Ask LLM for a short natural language summary
    summary_prompt = f"""
You are a hallucination analysis assistant.
Given the data below, write a 2-3 sentence plain English summary of how reliable the answer is.

Question: {query}
Reliability Score: {score} ({label})
Verdict Breakdown:
{verdict_block}

Rules:
- Be direct and honest
- Mention which claims failed if any
- No bullet points, just flowing sentences
- No robotic tone
- Keep it under 60 words
"""
    summary_response = llm.invoke(summary_prompt)
    summary = summary_response.content.strip()

    # Assemble final report
    report = f"""
╔══════════════════════════════════════════════════════╗
             HALLUCINATION RADAR — REPORT
╚══════════════════════════════════════════════════════╝

QUESTION : {query}

ANSWER   : {answer}

─────────────────────────────────────────────────────
 CLAIM VERDICTS
─────────────────────────────────────────────────────
{verdict_block}

─────────────────────────────────────────────────────
 RELIABILITY SCORE : {score} / 1.0  →  {label}
─────────────────────────────────────────────────────

 SUMMARY : {summary}

══════════════════════════════════════════════════════
"""

    return {"report": report}

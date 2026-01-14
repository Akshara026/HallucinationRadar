# HallucinationRadar
Hallucination Radar is a project that checks whether an AI (LLM) answer is actually true or just sounding true.

Sometimes chatbots give confident answers that are wrong (this is called hallucination).
This system acts like a truth-checker layer on top of AI.

What it does

When a user asks a question:

The AI generates an answer

The system breaks the answer into small factual statements (claims)

It searches trusted sources (like Wikipedia / PDFs / documents) for evidence

It checks each claim:

 Supported by evidence

 Partially supported / unsure

 Not supported (hallucination)

It gives a Truthfulness Score (0–100%)

It highlights risky sentences and shows citations
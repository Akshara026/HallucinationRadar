# HallucinationRadar

**HallucinationRadar** is a project that checks whether an AI (LLM) answer is actually true — or just *sounds* true.

Sometimes chatbots give confident answers that are wrong (**hallucinations**). HallucinationRadar acts like a **truth-checking layer** on top of AI responses.

---

## What it does

When a user asks a question, the system:

1. **Generates an AI answer**
2. **Breaks the answer into factual statements (claims)**
3. **Searches trusted sources** (e.g., Wikipedia, PDFs, verified documents) for evidence
4. **Evaluates each claim**:
   -  Supported by evidence
   -  Partially supported / unsure
   -  Not supported (hallucination)
5. **Outputs a Truthfulness Score** (0–100%)
6. **Highlights risky sentences** and provides **citations**

---

## Output

HallucinationRadar provides:

- **Truthfulness Score**: `0–100%`
- **Claim-by-claim verification table**
- **Risky/high-uncertainty sentence highlighting**
- **Citations** to supporting (or contradicting) sources

---

## Why it matters

LLMs can produce fluent answers that may still be incorrect. HallucinationRadar helps:

- Detect misinformation early
- Increase trust and transparency
- Make AI outputs verifiable and auditable

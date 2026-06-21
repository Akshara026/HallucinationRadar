import json
import re
from typing import Any, Dict, List

from langchain_ollama import ChatOllama

# Initializin the model once so it can be accescced easily by othr func
llm = ChatOllama(
    model="deepseek-r1:7b",
    temperature=0.6,
    num_ctx=8192,
    request_timeout=120,
)

JSON_SYSTEM_PROMPT = """
You are a JSON API.

Rules:
- Return only valid JSON.
- Do not output reasoning.
- Do not output analysis.
- Do not use markdown code fences.
- Do not include explanations or commentary.
- Return raw JSON only.
"""

FACT_DENSE_SYSTEM_PROMPT = """
You are a technical writer who produces fact-dense, verifiable content.
Every sentence you write must contain at least one specific, concrete fact
that can be independently verified. You write with precision and clarity.
"""


def clean_response(text: str) -> str:
    """Remove DeepSeek reasoning tags and other artifacts while preserving content."""

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL,
    ).strip()

    text = re.sub(r"```\w*\n", "", text)  # Opening fence
    text = re.sub(r"\n```", "", text)  # Closing fence

    text = "\n".join(line.strip() for line in text.split("\n"))  # removing whitespce

    # Remove multiple consecutive blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


def parse_json_array(text: str) -> List[str]:
    """Parse JSON array or object with 'concepts' key."""
    try:
        data = json.loads(text)

        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]

        if isinstance(data, dict):
            for key in ["concepts", "claims", "facts"]:
                if isinstance(data.get(key), list):
                    return [
                        str(item).strip() for item in data[key] if str(item).strip()
                    ]

    except json.JSONDecodeError:
        pass

    return []


def extract_factual_concepts(query: str) -> List[str]:
    """
    Extract specific, verifiable concepts and claims that need explanation.
    Focus on concrete elements rather than abstract topics.
    """
    prompt = f"""Identify the specific, verifiable facts and concepts needed to explain this topic.

Topic: {query}

Requirements for each concept:
- Must be a specific, concrete claim or fact
- Must be independently verifiable
- Must be precise (include numbers, dates, names where applicable)
- Must be technically accurate

Format: Return a JSON array of strings
Example for "TCP/IP protocol":
[
  "TCP/IP was developed by Vint Cerf and Bob Kahn in 1974 under DARPA funding",
  "TCP provides reliable, ordered delivery through sequence numbers and acknowledgments",
  "IP handles routing with 32-bit addresses in IPv4 and 128-bit in IPv6",
  "The three-way handshake uses SYN, SYN-ACK, and ACK packets to establish connections",
  "TCP congestion control algorithms include Tahoe (1988), Reno (1990), and CUBIC (2008)",
  "Maximum TCP segment size is typically 1460 bytes on Ethernet networks with MTU 1500",
  "TCP port numbers range from 0-65535, with 0-1023 reserved for well-known services",
  "IP fragmentation occurs when packet size exceeds path MTU, reassembled at destination",
  "The OSI model has 7 layers while TCP/IP uses 4 layers: Application, Transport, Internet, Link"
]

Return 7-12 specific, fact-dense claims."""

    messages = [
        ("system", JSON_SYSTEM_PROMPT),
        ("human", prompt),
    ]

    response = llm.invoke(messages)
    text = clean_response(response.content)
    concepts = parse_json_array(text)

    # Fallback if extraction fails
    if not concepts:
        concepts = [
            "Core technical specifications and standards",
            "Historical development timeline and key contributors",
            "Performance metrics and limitations",
            "Implementation details and algorithms",
            "Real-world applications and case studies",
            "Comparative analysis with alternatives",
        ]

    return concepts


def generate_fact_dense_answer(query: str, concepts: List[str]) -> str:
    """
    Generate an answer where every sentence contains a verifiable fact.
    No motivational context, no fluffy explanations, just concrete information.
    """
    concept_list = "\n".join(
        f"{i + 1}. {concept}" for i, concept in enumerate(concepts)
    )

    prompt = f"""Write a fact-dense technical explanation for this question:

Question: {query}

Cover these specific points (in this exact order):
{concept_list}

CRITICAL WRITING RULES:
1. EVERY sentence must contain at least one specific, verifiable fact
2. Use precise numbers, dates, names, and measurements whenever possible
3. No motivational statements (avoid: "This revolutionized...", "Interestingly...", "Importantly...")
4. No contextual filler (avoid: "To understand this...", "It is worth noting...", "One might wonder...")
5. No opinions or subjective assessments (avoid: "arguably", "perhaps", "many believe")
6. Define jargon immediately with precise definitions, not analogies
7. State limitations and failure modes explicitly with concrete examples
8. Each paragraph should contain 2-3 interconnected facts, not general statements
9. Write in plain text - no markdown, no numbering, no bullet points
10. Stop after the last fact - no summaries, no conclusions

BAD EXAMPLES (do not write like this):
"The transformer architecture revolutionized natural language processing."
"Interestingly, attention mechanisms allow models to focus on relevant parts."
"This approach has proven remarkably effective in various applications."

GOOD EXAMPLES (write like this):
"The transformer architecture, introduced by Vaswani et al. in the 2017 paper 'Attention Is All You Need', eliminated recurrence and convolution operations."
"Self-attention computes weighted representations using dot-product similarity with O(n²) complexity, where n is sequence length."
"BERT, released by Google in 2018, achieved 93.2% on SQuAD 1.1 using 340M parameters trained on BooksCorpus (800M words) and English Wikipedia (2,500M words)."

Your response:"""

    messages = [
        ("system", FACT_DENSE_SYSTEM_PROMPT),
        ("human", prompt),
    ]

    response = llm.invoke(messages)
    return clean_response(response.content)


def answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrate the fact-dense answer generation pipeline.
    """
    query = state["query"]

    # Extractin specific, fact-based concepts
    concepts = extract_factual_concepts(query)

    # Generatein fact-dense answer
    answer = generate_fact_dense_answer(
        query=query,
        concepts=concepts,
    )

    return {
        "concepts": concepts,
        "answer": answer,
    }


if __name__ == "__main__":
    state = {"query": "Can you tell me about transformers in machine learning?"}
    result = answer_node(state)

    print("\n" + "=" * 80)
    print("EXTRACTED FACTUAL CONCEPTS:")
    print("=" * 80)
    for i, concept in enumerate(result["concepts"], 1):
        print(f"{i}. {concept}")

    print("\n" + "=" * 80)
    print("FACT-DENSE ANSWER:")
    print("=" * 80)
    print(result["answer"])

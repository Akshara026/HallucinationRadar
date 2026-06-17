import json
import re

from langchain_ollama import ChatOllama

# Initializin the model once so it can be reused
llm = ChatOllama(
    model="deepseek-r1:7b",
    temperature=0.2,
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


def clean_response(text: str) -> str:
    """
    this just removes DeepSeek reasoning tags if they appear.
    """
    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL,
    ).strip()


def parse_json_array(text: str) -> list[str]:
    """
    Extract and validate a JSON array from model output.
    """
    match = re.search(r"\[[\s\S]*\]", text)

    if not match:
        return []

    try:
        data = json.loads(match.group())

        if isinstance(data, list):
            return [
                item.strip() for item in data if isinstance(item, str) and item.strip()
            ]

    except json.JSONDecodeError:
        pass

    return []


def extract_concepts(query: str) -> list[str]:
    """
    Extract the minimum set of concepts needed to explain a topic.
    """
    prompt = f"""
Identify the minimum concepts needed to explain the topic.

Question: {query}

Example:
[
  "Problem being solved",
  "Limitations of previous approaches",
  "Motivation for the new approach",
  "Core concepts",
  "Key components"
]

Rules:
- Return a JSON array of strings.
- Return 5 to 10 concepts maximum.
- Order concepts from beginner to advanced.
- Include limitations of previous approaches.
- Include motivation for the new approach.
"""

    messages = [
        ("system", JSON_SYSTEM_PROMPT),
        ("human", prompt),
    ]

    response = llm.invoke(messages)

    text = clean_response(response.content)

    concepts = parse_json_array(text)

    if not concepts or not all(isinstance(c, str) and c.strip() for c in concepts):
        concepts = [
            "Problem being solved",
            "Limitations of previous approaches",
            "Motivation for the new approach",
            "Core concepts",
            "Key components",
            "Applications and limitations",
        ]

    return concepts


def generate_answer(query: str, concepts: list[str]) -> str:
    """
    Generate a beginner-friendly explanation using extracted concepts.
    """
    concept_list = "\n".join(f"- {concept}" for concept in concepts)

    prompt = f"""
You are an expert teacher explaining technical topics to beginners.

Question:
{query}

Explain these concepts in order:

{concept_list}

Rules:
- Use numbered sections only.
- No markdown headings.
- No bullet points.
- No summary or conclusion.
- Start with the problem older approaches faced.
- Explain why older approaches failed.
- Introduce the new idea.
- Explain each concept in order.
- Define jargon immediately.
- Use analogies whenever possible.
- Focus on intuition before mathematics.
- Write naturally, as if teaching a curious beginner.
"""

    response = llm.invoke(prompt)

    return clean_response(response.content)


def answer_node(state: dict) -> dict:
    """
    Orchestrate the pipeline.
    """
    query = state["query"]

    concepts = extract_concepts(query)

    answer = generate_answer(
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

    print("\nConcepts:\n")

    for concept in result["concepts"]:
        print(f"- {concept}")

    print("\nAnswer:\n")
    print(result["answer"])


# import re

# from langchain_ollama import ChatOllama

# llm = ChatOllama(
#     model="deepseek-r1:7b",
#     temperature=0.2,
#     num_ctx=8192,
# )


# def answer_node(state):
#     query = state["query"]

#     prompt = f"""
# You are an expert teacher who explains technical topics to beginners.

# Rules:
# - If you are unsure, say "I don't know."
# - Never invent facts, examples, or sources.
# - Use simple, clear language.
# - Explain all important concepts.
# - Define technical terms when first introduced.
# - Use headings and bullet points.
# - Stay focused on the question.
# - Do not include a summary section.

# For educational topics, follow this exact structure:

# # Introduction
# Briefly introduce the topic.

# # What is it?
# Provide a simple definition.

# # Why was it created?
# Explain the limitations of previous approaches and the problem this topic solves.

# # Core Concepts
# List and explain each important concept separately.

# # How It Works
# Describe the process step by step.

# # Example or Analogy
# Give an intuitive example.

# # Applications
# Explain where it is used.

# # Limitations
# Mention drawbacks and challenges.

# # Further Learning
# Suggest related topics to explore.

# Before writing the answer:

# 1. Identify the essential concepts needed to answer the question.
# 2. Ensure every concept is covered.
# 3. Organize concepts from basic to advanced.
# 4. Then write the final answer buty dont add those headings.

# Question: {query}
# """

#     response = llm.invoke(prompt)

#     answer = response.content

#     # Removes reasoning tags from DeepSeek-R1
#     answer = re.sub(
#         r"<think>.*?</think>",
#         "",
#         answer,
#         flags=re.DOTALL,
#     ).strip()

#     return {"answer": answer}


# if __name__ == "__main__":
#     state = {"query": "Can you tell me about transformers in machine learning?"}
#     result = answer_node(state)
#     print(result["answer"])

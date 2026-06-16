import json
import re

from langchain_ollama import ChatOllama

# initisalising model once so it can be accesed multple func at same time
llm = ChatOllama(
    model="deepseek-r1:7b",
    temperature=0.2,
    num_ctx=8192,
)


def clean_response(text: str) -> str:
    """
    this func to remove DeepSeek reasoning tags.
    """
    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL,
    ).strip()


def extract_concepts(query: str) -> list[str]:
    """
    Call the LLM to extract concepts from the user's question.
    """

    prompt = f"""
Identify the minimum set of concepts needed to teach this topic.

Question: {query}

Rules:
- Return ONLY a JSON array.
- Order concepts from beginner to advanced.
- Include the limitations of previous approaches.
- Include the motivation for the new approach.
- Exclude implementation details unless essential.
- Return 5 to 10 concepts maximum.
"""

    response = llm.invoke(prompt)

    text = clean_response(response.content)

    try:
        concepts = json.loads(text)

        if isinstance(concepts, list):
            return concepts

    except json.JSONDecodeError:
        pass

    # Fallback for invalid JSON
    return [line.strip("- ").strip() for line in text.splitlines() if line.strip()]


def generate_answer(query: str, concepts: list[str]) -> str:
    """
    callin LLM again to generate the ans using the concepts tht we ectrcted abv
    :3
    """

    concept_list = "\n".join(f"- {concept}" for concept in concepts)

    prompt = f"""
    You are an expert teacher explaining technical topics to beginners.

    Question:
    {query}

    You MUST explain all of these concepts:

    {concept_list}

    Teaching style requirements:

    - Teach through a story, not a textbook.
    - Start with the problem that existed before this technology.
    - Explain why previous approaches failed.
    - Introduce the new solution.
    - Explain concepts in the given order.
    - Build from simple ideas to advanced ones.
    - Use analogies whenever possible.
    - Explain jargon immediately after introducing it.
    - Focus on intuition before mathematical details.
    - Use numbered sections.
    - Write naturally, as if teaching a curious beginner.
    - Do NOT use headings like "Introduction", "Core Concepts", or "Applications".
    - Do NOT add a summary or conclusion.

    Recommended flow:

    1. The problem with older approaches
    2. The new idea
    3. How the core mechanism works
    4. The major components
    5. Why this was a breakthrough
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

    print("\n concepts :3\n")
    for concept in result["concepts"]:
        print(f"- {concept}")

    print("\n main answer :3\n")
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

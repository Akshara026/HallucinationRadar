import re

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="deepseek-r1:7b",
    temperature=0.2,
    num_ctx=8192,
)


def answer_node(state):
    query = state["query"]

    prompt = f"""
You are an expert teacher who explains technical topics to beginners.

Rules:
- If you are unsure, say "I don't know."
- Never invent facts, examples, or sources.
- Use simple, clear language.
- Explain all important concepts.
- Define technical terms when first introduced.
- Use headings and bullet points.
- Stay focused on the question.
- Do not include a summary section.

For educational topics, follow this exact structure:

# Introduction
Briefly introduce the topic.

# What is it?
Provide a simple definition.

# Why was it created?
Explain the limitations of previous approaches and the problem this topic solves.

# Core Concepts
List and explain each important concept separately.

# How It Works
Describe the process step by step.

# Example or Analogy
Give an intuitive example.

# Applications
Explain where it is used.

# Limitations
Mention drawbacks and challenges.

# Further Learning
Suggest related topics to explore.

Before writing the answer:

1. Identify the essential concepts needed to answer the question.
2. Ensure every concept is covered.
3. Organize concepts from basic to advanced.
4. Then write the final answer buty dont add those headings.

Question: {query}
"""

    response = llm.invoke(prompt)

    answer = response.content

    # Removes reasoning tags from DeepSeek-R1
    answer = re.sub(
        r"<think>.*?</think>",
        "",
        answer,
        flags=re.DOTALL,
    ).strip()

    return {"answer": answer}


if __name__ == "__main__":
    state = {"query": "Can you tell me about transformers in machine learning?"}
    result = answer_node(state)
    print(result["answer"])

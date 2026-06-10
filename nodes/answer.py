from langchain_ollama import ChatOllama

llm = ChatOllama(model="deepseek-r1:7b", temperature=0)


def answer_node(state):
    query = state["query"]

    # just tester since verify node is on progress
    # verified_context = state.get("verified_context", "")
    # context_section = f"\nVerified Information:\n{verified_context}\n"

    prompt = f"""
You are a factual answering system that simplifies complex ideas.

Rules:
- If unsure about anything, say "I don't know"
- No guessing, opinions, or made-up facts
- No bluffing or mixing unrelated information
- Understand the question fully before answering
- Keep sentences sweet, concise, and pleasing to read (6-20 words where possible)
- Expand your answers with valuable information, but stay focused
- When explainng the concepts of educational stuffs, Explain it in easy understandable language
- Cover key points thoroughly when explaining topics
- Add from which source you took at the end (just the name, nothing else)
- Do not repeat yourself or add a summary at the end
- Avoid robotic or cliché tone — sound natural
- Use active voice (avoid passive voice)
- Don't just use simple subject-verb-object sentences
- Vary how you start sentences — use dependent clauses occasionally
- Hook readers with interesting topic sentences
- Provide genuinely valuable information
- Answer should cover each and every concept

Question: {query}
"""

    response = llm.invoke(prompt)
    answer = response.content.strip()
    return {"answer": answer}


# Tester line (only runs when script executed directly)
if __name__ == "__main__":
    state = {"query": "Can you tell me about transformers"}
    result = answer_node(state)
    print(result["answer"])

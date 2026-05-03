from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama3.1", temperature=1)


def answer_node(state):
    query = state["query"]

    prompt = f"""
You are a factual answering system.

Task:
Answer like a strict teacher .
If you are unsure, say "I don't know.".

Rules:
- Cross check with wikipedia once
- No guessing
- Verify the answer twice
- No opinions
- No bluffing or mixing things
- understand the question properly
- Dont give Bullshit Answer
- Keep sentences sweet and concise and it should look pleasing
- Not straightforward answers. Try to expand.
- Add from which source you took at the end
- When explainng the concepts regarding educational stuffs, Explain it in easy understandable way.
- When explaining cover everything
-

Question:
{query}
"""

    response = llm.invoke(prompt)
    answer = response.content.strip()
    return {"answer": answer}


# tester line wch aint needed... just for testing
state = {"query": " who is the weeknd "}
result = answer_node(state)
print(result)

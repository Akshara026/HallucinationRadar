from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama3.1", temperature=0)


def claims_node(state):
    answer = state["answer"]

    prompt = f"""
Extract atomic factual claims from the text below.

Rules:
- One claim per line
- Each claim must contain only one fact
- No opinions
- No combined sentences
- Claims source should be from origin
- Keep claims clear and concise

Text:
{answer}
"""

    response = llm.invoke(prompt)

    claims = [c.strip("- ").strip() for c in response.content.split("\n") if c.strip()]

    return {"claims": claims}

import wikipedia
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama3.1", temperature=0)


def retrieval_node(state):
    claims = state["claims"]
    evidence = {}

    for claim in claims:
        # Ask the LLM to form a tight search query for this claim
        query_prompt = f"""
Convert this claim into a short Wikipedia search query (5 words max):
Claim: {claim}
Query:
"""
        query_response = llm.invoke(query_prompt)
        search_query = query_response.content.strip()

        # Fetch Wikipedia snippets
        snippets = []
        try:
            results = wikipedia.search(search_query, results=3)
            for title in results[:2]:  # top 2 articles only
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    # Grab the first 500 chars as evidence chunk
                    snippets.append(page.content[:500])
                except Exception:
                    continue
        except Exception:
            snippets.append(" nothing found apparently T_T . ggs mate Y_Y")

        evidence[claim] = snippets

    return {"evidence": evidence}

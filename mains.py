from nodes.answer import answer_node
from nodes.claims import claims_node


def run_pipeline(query):
    state = {"query": query}

    # Step 1: Generate answer
    state.update(answer_node(state))

    # Step 2: Extract claims
    state.update(claims_node(state))

    return state


if __name__ == "__main__":
    result = run_pipeline("What is an LLM?")

    print("\n--- ANSWER ---\n")
    print(result["answer"])

    print("\n--- CLAIMS ---\n")
    for c in result["claims"]:
        print("-", c)

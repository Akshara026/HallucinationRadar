from nodes.answer import answer_node
from nodes.claims import claims_node


def run_pipeline(query):
    state = {"query": query}

    state.update(answer_node(state))  # generatin ans

    state.update(claims_node(state))  # extractin claims

    return state


if __name__ == "__main__":
    result = run_pipeline("What is an LLM?")

    print("\n ANSWER :3\n")
    print(result["answer"])

    print("\n CLAIMS :3\n")
    for c in result["claims"]:
        print("-", c)

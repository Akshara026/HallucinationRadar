from nodes.answer import answer_node
from nodes.claims import claims_node
from nodes.retrieval import retrieval_node
from nodes.verify import verify_node


def run_pipeline(query):
    state = {"query": query}

    state.update(answer_node(state))
    state.update(claims_node(state))
    state.update(retrieval_node(state))
    state.update(verify_node(state))

    return state


if __name__ == "__main__":
    result = run_pipeline("What is an LLM?")

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "=" * 60)
    print("CLAIMS")
    print("=" * 60)
    for i, c in enumerate(result["claims"], 1):
        print(f"{i}. {c}")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    for claim, verdict in result["verdicts"].items():
        print(f"\nClaim: {claim}")
        print(f"Verdict: {verdict['verdict']} ({verdict['confidence']:.1%})")
        print(f"Reasoning: {verdict['reasoning'][:200]}...")

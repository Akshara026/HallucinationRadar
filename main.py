from nodes.answer import answer_node
from nodes.claims import claims_node
from nodes.retrieval import retrieval_node


def run_pipeline(query):
    state = {"query": query}

    state.update(answer_node(state))

    state.update(claims_node(state))

    state.update(retrieval_node(state))

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
    print("EVIDENCE")
    print("=" * 60)
    for claim, evidence_list in result["evidence"].items():
        print(f"\nClaim: {claim}")
        print("-" * 40)
        for ev in evidence_list:
            print(f"  [{ev.get('relevance', 0):.1%}] {ev['title']}")
            print(f"  {ev['content'][:150]}...")

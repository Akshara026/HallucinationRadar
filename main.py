from nodes.answer import answer_node
from nodes.claims import claims_node
from nodes.report import report_node, save_report
from nodes.retrieval import retrieval_node
from nodes.score import score_node
from nodes.verify import verify_node


def run_pipeline(query, save=True):
    state = {"query": query}

    # Step 1: Generate answer
    state.update(answer_node(state))
    print(f"\nDEBUG: answer length = {len(state['answer'])}")
    print(f"DEBUG: first 300 chars = {state['answer'][:300]}")

    # Step 2: Extract claims
    state.update(claims_node(state))
    print(f"DEBUG: claims count = {len(state['claims'])}")
    for i, c in enumerate(state["claims"], 1):
        print(f"DEBUG: claim {i}: {c[:100]}...")

    # Step 3: Retrieve evidence
    state.update(retrieval_node(state))

    # Step 4: Verify claims
    state.update(verify_node(state))

    # Step 5: Score hallucination
    state.update(score_node(state))

    # Step 6: Generate report
    state.update(report_node(state))

    # Save report to file
    if save:
        save_report(state["report"])

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

    print("\n" + "=" * 60)
    print("HALLUCINATION SCORE")
    print("=" * 60)
    score_data = result["score"]
    print(f"Overall Score: {score_data['overall_score']:.3f}")
    print(f"Verdict: {score_data['verdict']}")
    print(f"Risk Level: {score_data['hallucination_risk']}")
    print(f"\nBreakdown:")
    b = score_data["breakdown"]
    print(f"  Total Claims: {b['total']}")
    print(f"  ✅ Supported: {b['supported']}")
    print(f"  ❌ Contradicted: {b['contradicted']}")
    print(f"  ❓ Unverifiable: {b['unverifiable']}")
    print(f"  ⚠️  Insufficient Evidence: {b['insufficient_evidence']}")

    if score_data["high_risk_claims"]:
        print(f"\n⚠️  High Risk Claims (likely hallucinations):")
        for claim in score_data["high_risk_claims"]:
            print(f"  - {claim['claim'][:120]}... ({claim['confidence']:.1%})")

    if score_data["verified_claims"]:
        print(f"\n✅ Verified Claims:")
        for claim in score_data["verified_claims"]:
            print(f"  - {claim['claim'][:120]}... ({claim['confidence']:.1%})")

    print("\n" + "=" * 60)
    print("FULL REPORT")
    print("=" * 60)
    print(result["report"])

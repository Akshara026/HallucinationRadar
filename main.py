"""
main.py - HallucinationRadar entry point
Uses LangGraph pipeline for execution
"""

from graph import run_pipeline
from nodes.report import save_report_pdf


if __name__ == "__main__":
    query = "What is an LLM?"

    print("=" * 60)
    print("HALLUCINATION RADAR")
    print("=" * 60)
    print(f"\nQuery: {query}")
    print("\nStarting pipeline...\n")

    # Run the graph pipeline
    result = run_pipeline(query)

    # Save PDF report
    report_path = save_report_pdf(result["report"])

    # Print final output
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

    print(f"\nPDF Report saved to: {report_path}")

"""
main.py - HallucinationRadar entry point

Supports two modes:
1. Query mode: python3 main.py query "What is an LLM?"
2. External mode: python3 main.py external "Pasted text to check..."
"""

import sys
from graph import run_pipeline
from nodes.report import save_report_pdf


def check_query(query: str):
    """Run pipeline with a query (generates answer first)."""
    print("=" * 60)
    print("HALLUCINATION RADAR - QUERY MODE")
    print("=" * 60)
    print(f"\nQuery: {query}")
    print("\nStarting pipeline...\n")

    result = run_pipeline(query=query)
    return result


def check_external_answer(text: str, context: str = ""):
    """Run pipeline with external text (skips answer generation)."""
    print("=" * 60)
    print("HALLUCINATION RADAR - EXTERNAL CHECK MODE")
    print("=" * 60)
    print(f"\nChecking external text ({len(text)} chars)")
    if context:
        print(f"Context: {context}")
    print("\nStarting pipeline...\n")

    result = run_pipeline(query=context, external_answer=text)
    return result


def print_results(result):
    """Print pipeline results."""
    report_path = save_report_pdf(result["report"])

    print("\n" + "=" * 60)
    print(f"ANSWER (source: {result.get('answer_source', 'unknown')})")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "=" * 60)
    print("CLAIMS")
    print("=" * 60)
    for i, c in enumerate(result.get("claims", []), 1):
        print(f"{i}. {c}")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    for claim, verdict in result.get("verdicts", {}).items():
        print(f"\nClaim: {claim}")
        print(f"Verdict: {verdict['verdict']} ({verdict['confidence']:.1%})")
        print(f"Reasoning: {verdict['reasoning'][:200]}...")

    print("\n" + "=" * 60)
    print("HALLUCINATION SCORE")
    print("=" * 60)
    score_data = result.get("score", {})
    print(f"Overall Score: {score_data.get('overall_score', 0):.3f}")
    print(f"Verdict: {score_data.get('verdict', 'UNKNOWN')}")
    print(f"Risk Level: {score_data.get('hallucination_risk', 'UNKNOWN')}")
    print(f"\nBreakdown:")
    b = score_data.get("breakdown", {})
    print(f"  Total Claims: {b.get('total', 0)}")
    print(f"  ✅ Supported: {b.get('supported', 0)}")
    print(f"  ❌ Contradicted: {b.get('contradicted', 0)}")
    print(f"  ❓ Unverifiable: {b.get('unverifiable', 0)}")
    print(f"  ⚠️  Insufficient Evidence: {b.get('insufficient_evidence', 0)}")

    if score_data.get("high_risk_claims"):
        print(f"\n⚠️  High Risk Claims (likely hallucinations):")
        for claim in score_data["high_risk_claims"]:
            print(f"  - {claim['claim'][:120]}... ({claim['confidence']:.1%})")

    if score_data.get("verified_claims"):
        print(f"\n✅ Verified Claims:")
        for claim in score_data["verified_claims"]:
            print(f"  - {claim['claim'][:120]}... ({claim['confidence']:.1%})")

    print("\n" + "=" * 60)
    print("FULL REPORT")
    print("=" * 60)
    print(result["report"])

    print(f"\n📄 PDF Report saved to: {report_path}")

    return result


if __name__ == "__main__":
    # Default: query mode with photosynthesis
    if len(sys.argv) < 2:
        result = check_query("How does photosynthesis work?")
        print_results(result)

    # Query mode: python3 main.py query "What is an LLM?"
    elif sys.argv[1].lower() == "query":
        query = sys.argv[2] if len(sys.argv) > 2 else "What is an LLM?"
        result = check_query(query)
        print_results(result)

    # External mode: python3 main.py external "Pasted text here..."
    elif sys.argv[1].lower() == "external":
        if len(sys.argv) > 2:
            text = sys.argv[2]
            context = sys.argv[3] if len(sys.argv) > 3 else ""
            result = check_external_answer(text, context)
            print_results(result)
        else:
            print("Usage: python3 main.py external \"Text to check...\" [context]")
            print("\nExample:")
            print('  python3 main.py external "GPT-4 has 100 trillion parameters" "Checking GPT-4 claims"')

    else:
        print("Usage:")
        print("  python3 main.py                          # Default query mode")
        print('  python3 main.py query "Your question"    # Query mode')
        print('  python3 main.py external "Text" "Context" # External check mode')

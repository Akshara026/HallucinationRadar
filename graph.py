"""
graph.py - LangGraph pipeline for HallucinationRadar

Supports two modes:
1. Query mode: query → answer → claims → retrieval → verify → score → report
2. External mode: external_answer → claims → retrieval → verify → score → report
"""

from langgraph.graph import StateGraph, END
from state import RadarState

from nodes.prepare_input import prepare_input_node, should_generate_answer
from nodes.answer import answer_node
from nodes.claims import claims_node
from nodes.retrieval import retrieval_node
from nodes.verify import verify_node
from nodes.score import score_node
from nodes.report import report_node


def build_graph():
    """Build and compile the LangGraph pipeline with conditional routing."""

    graph = StateGraph(RadarState)

    # Add all nodes
    graph.add_node("prepare_input", prepare_input_node)
    graph.add_node("answer", answer_node)
    graph.add_node("claims", claims_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("verify", verify_node)
    graph.add_node("score", score_node)
    graph.add_node("report", report_node)

    # Entry point is prepare_input
    graph.set_entry_point("prepare_input")

    # Conditional routing:
    # - If external_answer provided → skip answer_node, go to claims
    # - Otherwise → generate answer first
    graph.add_conditional_edges(
        "prepare_input",
        should_generate_answer,
        {
            "claims": "claims",  # Skip answer generation
            "answer": "answer",   # Generate answer first
        }
    )

    # Linear flow after claims
    graph.add_edge("answer", "claims")
    graph.add_edge("claims", "retrieval")
    graph.add_edge("retrieval", "verify")
    graph.add_edge("verify", "score")
    graph.add_edge("score", "report")
    graph.add_edge("report", END)

    # Compile and return
    return graph.compile()


# Create pipeline instance
pipeline = build_graph()


def run_pipeline(query: str = "", external_answer: str = ""):
    """
    Run the pipeline with either:
    - query: Generate an answer and check it
    - external_answer: Check pasted text directly (skips answer generation)
    """
    input_state = {}

    if external_answer and external_answer.strip():
        input_state["external_answer"] = external_answer.strip()
        if query:
            input_state["query"] = query
    elif query:
        input_state["query"] = query
    else:
        raise ValueError("Must provide either query or external_answer")

    result = pipeline.invoke(input_state)
    return result


if __name__ == "__main__":
    # Test query mode
    print("=" * 60)
    print("TESTING QUERY MODE")
    print("=" * 60)
    result = run_pipeline(query="What is an LLM?")
    print(f"\nAnswer source: {result.get('answer_source', 'unknown')}")
    print(f"Answer length: {len(result.get('answer', ''))}")
    print(f"Claims: {len(result.get('claims', []))}")
    print(f"Score: {result.get('score', {}).get('overall_score', 0)}")

    print("\n" + "=" * 60)
    print("TESTING EXTERNAL MODE")
    print("=" * 60)
    external_text = """GPT-4 was developed by OpenAI and released in 2023.
    It has 1.8 trillion parameters and was trained on data up to 2022."""
    result2 = run_pipeline(external_answer=external_text)
    print(f"\nAnswer source: {result2.get('answer_source', 'unknown')}")
    print(f"Answer length: {len(result2.get('answer', ''))}")
    print(f"Claims: {len(result2.get('claims', []))}")
    print(f"Score: {result2.get('score', {}).get('overall_score', 0)}")

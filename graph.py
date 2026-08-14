"""
graph.py - LangGraph pipeline for HallucinationRadar

Connects all nodes in a linear pipeline:
query → answer → claims → retrieval → verify → score → report
"""

from langgraph.graph import StateGraph, END
from state import RadarState

from nodes.answer import answer_node
from nodes.claims import claims_node
from nodes.retrieval import retrieval_node
from nodes.verify import verify_node
from nodes.score import score_node
from nodes.report import report_node


def build_graph():
    """Build and compile the LangGraph pipeline."""

    # Create graph with our state type
    graph = StateGraph(RadarState)

    # Add all nodes
    graph.add_node("answer", answer_node)
    graph.add_node("claims", claims_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("verify", verify_node)
    graph.add_node("score", score_node)
    graph.add_node("report", report_node)

    # Define the flow (linear pipeline)
    graph.set_entry_point("answer")
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


def run_pipeline(query: str):
    """Run the full pipeline with a query."""
    result = pipeline.invoke({"query": query})
    return result


if __name__ == "__main__":
    # Test the graph
    result = run_pipeline("What is an LLM?")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nAnswer length: {len(result.get('answer', ''))}")
    print(f"Claims extracted: {len(result.get('claims', []))}")
    print(f"Verdicts: {len(result.get('verdicts', {}))}")
    print(f"Score: {result.get('score', {}).get('overall_score', 0)}")
    print(f"Report length: {len(result.get('report', ''))}")

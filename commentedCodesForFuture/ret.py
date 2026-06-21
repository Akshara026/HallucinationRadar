# this contain the actual code but due to machine constraint, we r modifying the code and adding it in retrival node :3
# sorry for the inconvience

"""
retrieval.py - Evidence Retrieval Node

Smart Wikipedia evidence retrieval using:
- qwen2.5:7b for query generation
- nomic-embed-text for relevance scoring
- Proper error handling and fallbacks
"""

import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
import wikipedia
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Initialize models once at module level
llm = ChatOllama(model="qwen2.5:7b", temperature=0, num_ctx=4096)
embeddings = OllamaEmbeddings(model="nomic-embed-text")


def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve Wikipedia evidence for each claim.

    Input:  state["claims"] - list of factual claims
    Output: state["evidence"] - dict mapping claim → list of evidence dicts
    """
    claims = state.get("claims", [])
    evidence = {}

    for i, claim in enumerate(claims):
        print(f"Retrieving evidence {i + 1}/{len(claims)}: {claim[:80]}...")
        evidence[claim] = retrieve_for_claim(claim)

    return {"evidence": evidence}


def retrieve_for_claim(claim: str) -> List[Dict[str, str]]:
    """
    Full retrieval pipeline for a single claim.
    Returns list of evidence dicts with title, content, url, and relevance score.
    """
    # Step 1: Generate smart search queries
    queries = generate_queries(claim)

    # Step 2: Fetch Wikipedia articles for all queries
    all_articles = []
    for query in queries:
        articles = fetch_wikipedia_articles(query)
        all_articles.extend(articles)

    # Step 3: Deduplicate articles
    unique_articles = deduplicate_articles(all_articles)

    # Step 4: Extract relevant chunks from articles
    chunks = extract_relevant_chunks(claim, unique_articles)

    # Step 5: Score chunks by relevance to claim
    scored_chunks = score_chunks(claim, chunks)

    # Step 6: Select top evidence
    top_evidence = scored_chunks[:3] if scored_chunks else []

    # Step 7: Fallback if nothing found
    if not top_evidence:
        top_evidence = [
            {
                "title": "No evidence found",
                "content": f"No Wikipedia articles found for: {claim}",
                "url": "",
                "relevance": 0.0,
            }
        ]

    return top_evidence


def generate_queries(claim: str) -> List[str]:
    """
    Use LLM to generate multiple search queries for better coverage.
    """
    prompt = f"""Generate 3 different Wikipedia search queries to find evidence for this claim.
Each query should take a different approach.

Claim: "{claim}"

Return ONLY a JSON array: ["query1", "query2", "query3"]

Rules:
- Query 1: Search for the main entity/fact directly
- Query 2: Search for related context/background
- Query 3: Search for specific details mentioned (dates, numbers, names)
- Each query max 7 words
- No punctuation in queries
- Return ONLY the JSON array, nothing else"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Extract JSON array from response
        json_match = re.search(r"\[.*?\]", content, re.DOTALL)
        if json_match:
            queries = json.loads(json_match.group(0))
            if isinstance(queries, list) and len(queries) > 0:
                return [q.strip() for q in queries if q.strip()][:3]
    except Exception as e:
        print(f"  Query generation failed: {e}")

    # Fallback: Use claim itself and variations
    words = claim.split()
    clean = " ".join([w for w in words if len(w) > 3][:6])
    return [clean] if clean else [claim[:60]]


def fetch_wikipedia_articles(query: str, max_articles: int = 3) -> List[Dict[str, Any]]:
    """
    Fetch Wikipedia articles for a search query.
    Handles disambiguation, page errors, and redirects.
    """
    articles = []

    try:
        # Search Wikipedia
        results = wikipedia.search(query, results=max_articles)

        if not results:
            return articles

        for title in results:
            try:
                page = wikipedia.page(title, auto_suggest=False)

                # Extract meaningful content
                articles.append(
                    {
                        "title": page.title,
                        "url": page.url,
                        "summary": page.summary,
                        "content": page.content,
                        "page_id": page.pageid,
                    }
                )

            except wikipedia.exceptions.DisambiguationError as e:
                # Try first disambiguation option
                if e.options and len(e.options) > 0:
                    try:
                        page = wikipedia.page(e.options[0], auto_suggest=False)
                        articles.append(
                            {
                                "title": page.title,
                                "url": page.url,
                                "summary": page.summary,
                                "content": page.content,
                                "page_id": page.pageid,
                            }
                        )
                    except:
                        continue

            except (wikipedia.exceptions.PageError, wikipedia.exceptions.RedirectError):
                continue

            except Exception as e:
                print(f"  Error fetching '{title}': {str(e)[:50]}")
                continue

    except Exception as e:
        print(f"  Search failed for '{query}': {str(e)[:50]}")

    return articles


def deduplicate_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate articles by page_id.
    """
    seen_ids = set()
    unique = []

    for article in articles:
        if article["page_id"] not in seen_ids:
            seen_ids.add(article["page_id"])
            unique.append(article)

    return unique


def extract_relevant_chunks(
    claim: str, articles: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Extract relevant text chunks from Wikipedia articles.
    Uses summary + first few paragraphs.
    """
    chunks = []

    for article in articles:
        # Always include summary (concise overview)
        summary = article.get("summary", "")
        if summary and len(summary) > 50:
            chunks.append(
                {
                    "title": article["title"],
                    "url": article["url"],
                    "content": summary[:800].strip(),
                    "source_type": "summary",
                }
            )

        # Extract first few paragraphs (skip headers, empty lines)
        content = article.get("content", "")
        paragraphs = content.split("\n\n")

        relevant_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            # Skip section headers, empty paras, very short paras
            if not para or para.startswith("==") or len(para) < 100:
                continue
            relevant_paragraphs.append(para)

        # Take first 2 relevant paragraphs
        for para in relevant_paragraphs[:2]:
            chunks.append(
                {
                    "title": article["title"],
                    "url": article["url"],
                    "content": para[:800].strip(),
                    "source_type": "paragraph",
                }
            )

    return chunks


def score_chunks(claim: str, chunks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Score chunks by semantic similarity to claim using embeddings.
    Returns chunks sorted by relevance.
    """
    if not chunks:
        return []

    try:
        # Get claim embedding
        claim_embedding = np.array(embeddings.embed_query(claim))

        scored = []
        for chunk in chunks:
            # Get chunk embedding
            chunk_embedding = np.array(embeddings.embed_query(chunk["content"][:500]))

            # Cosine similarity
            similarity = np.dot(claim_embedding, chunk_embedding) / (
                np.linalg.norm(claim_embedding) * np.linalg.norm(chunk_embedding)
            )

            scored.append(
                {
                    "title": chunk["title"],
                    "url": chunk["url"],
                    "content": chunk["content"],
                    "relevance": round(float(similarity), 4),
                    "source_type": chunk["source_type"],
                }
            )

        # Sort by relevance descending
        scored.sort(key=lambda x: x["relevance"], reverse=True)

        return scored

    except Exception as e:
        print(f"  Scoring failed: {e}")
        # Return unscored chunks
        return [
            {
                "title": c["title"],
                "url": c["url"],
                "content": c["content"],
                "relevance": 0.0,
                "source_type": c.get("source_type", "unknown"),
            }
            for c in chunks
        ]


# Quick test
if __name__ == "__main__":
    test_state = {
        "claims": [
            "The Eiffel Tower was completed in 1889",
            "Python was created by Guido van Rossum in 1991",
            "Mount Everest is the tallest mountain on Earth",
            "Water boils at 100 degrees Celsius at sea level",
        ]
    }

    result = retrieval_node(test_state)

    print("\n" + "=" * 70)
    print("RETRIEVED EVIDENCE")
    print("=" * 70)

    for claim, evidence_list in result["evidence"].items():
        print(f"\n📌 Claim: {claim}")
        print("-" * 50)

        for i, ev in enumerate(evidence_list, 1):
            print(f"\n  Source {i}: {ev['title']}")
            print(f"  Relevance: {ev['relevance']:.2%}")
            print(f"  Content: {ev['content'][:150]}...")
            if ev.get("url"):
                print(f"  URL: {ev['url']}")

"""
so this isnt actuall retrieval node. the below is downgraded version for lower end machine due to machibne constraint T_T.
(Evidence Retrieval Node)

this is optimized for GTX 1650 hardware:
- Batched embedding calls
- Article caching across claims
- Reduced API calls
- Defensive error handling

for more info pls go to end of this code :3
"""

import hashlib
import json
import re
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import wikipedia
from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(
    model="qwen2.5:7b", temperature=0, num_ctx=2048
)  # context window has reduced (lower model lap T_T)
embeddings = OllamaEmbeddings(model="nomic-embed-text")


# Module-level cache to avoid re-fetching same articles
_article_cache: Dict[str, Dict[str, Any]] = {}
_embedding_cache: Dict[str, np.ndarray] = {}


def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve Wikipedia evidence for all claims.
    Uses caching and batching for performance.

    Input:  state["claims"] - list of factual claims
    Output: state["evidence"] - dict mapping claim → list of evidence dicts
    """
    claims = state.get("claims", [])

    if not claims:
        return {"evidence": {}}

    start_time = time.time()

    print(f"\nGenerating search queries for {len(claims)} claims...")
    all_queries = generate_all_queries(claims)  # One LLM call for all claims

    print("Fetching Wikipedia articles...")
    articles = collect_all_articles(all_queries)

    print("Extracting text chunks...")
    all_chunks = extract_all_chunks(claims, articles)

    # Step 4: Batch embed all chunks + claims (one embedding call where possible)
    print("🧮 Computing relevance scores...")
    scored_evidence = batch_score_chunks(claims, all_chunks)

    elapsed = time.time() - start_time
    print(f"✅ Retrieval complete in {elapsed:.1f}s")

    return {"evidence": scored_evidence}


def generate_all_queries(claims: List[str]) -> Dict[str, List[str]]:
    """
    Generate search queries for ALL claims in ONE LLM call.
    This is the biggest performance win - 1 call instead of N calls.
    """
    claims_text = "\n".join([f"{i + 1}. {claim}" for i, claim in enumerate(claims)])

    prompt = f"""For each claim below, generate 2 Wikipedia search queries to find supporting or refuting evidence.

Claims:
{claims_text}

Return a JSON object mapping claim number to its queries:
{{
    "1": ["query1", "query2"],
    "2": ["query1", "query2"]
}}

Rules:
- Each query max 6 words
- Focus on key entities, dates, names
- No punctuation
- Return ONLY the JSON object"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Extract JSON
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            queries_map = json.loads(json_match.group(0))

            # Map back to claims
            result = {}
            for i, claim in enumerate(claims):
                key = str(i + 1)
                if key in queries_map:
                    result[claim] = queries_map[key][:2]
                else:
                    # Fallback for this claim
                    result[claim] = [clean_query_fallback(claim)]
            return result
    except Exception as e:
        print(f"  Batch query generation failed: {e}")

    # Complete fallback
    return {claim: [clean_query_fallback(claim)] for claim in claims}


def clean_query_fallback(claim: str) -> str:
    """Simple keyword extraction fallback."""
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "was",
        "were",
        "are",
        "been",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
    }

    words = re.sub(r"[^\w\s]", " ", claim).split()
    key_terms = [w for w in words if w.lower() not in stop_words and len(w) > 2]
    return " ".join(key_terms[:5]) if key_terms else claim[:50]


def collect_all_articles(
    all_queries: Dict[str, List[str]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch ALL unique Wikipedia articles needed.
    Uses module-level cache to avoid re-fetching.
    """
    # Collect all unique queries
    unique_queries = set()
    for queries in all_queries.values():
        unique_queries.update(queries)

    # Fetch articles for each unique query (with caching)
    query_articles = {}
    for query in unique_queries:
        query_articles[query] = fetch_wikipedia_articles(query)

    # Map claims to their articles
    claim_articles = {}
    for claim, queries in all_queries.items():
        articles_for_claim = []
        seen_ids = set()

        for query in queries:
            for article in query_articles.get(query, []):
                page_id = article.get("page_id")
                if page_id and page_id not in seen_ids:
                    seen_ids.add(page_id)
                    articles_for_claim.append(article)

        claim_articles[claim] = articles_for_claim[:3]  # Max 3 articles per claim

    return claim_articles


def fetch_wikipedia_articles(query: str) -> List[Dict[str, Any]]:
    """
    Fetch Wikipedia articles with caching.
    Uses module-level cache keyed by query.
    """
    cache_key = query.lower().strip()

    if cache_key in _article_cache:
        return _article_cache[cache_key]

    articles = []

    try:
        results = wikipedia.search(query, results=2)  # Reduced from 3 to 2

        for title in results:
            try:
                page = _fetch_single_page(title)
                if page:
                    articles.append(page)

            except Exception:
                continue

    except Exception as e:
        print(f"  Search failed for '{query}': {str(e)[:50]}")

    _article_cache[cache_key] = articles
    return articles


def _fetch_single_page(title: str) -> Dict[str, Any]:
    """
    Safely fetch a single Wikipedia page with defensive error handling.
    """
    # Check cache first
    if title in _article_cache:
        return _article_cache[title]

    try:
        page = wikipedia.page(title, auto_suggest=False)
        return {
            "title": page.title,
            "url": page.url,
            "summary": page.summary[:500],  # Only store what we need
            "content_first_para": _get_first_paragraphs(page.content, num_paras=2),
            "page_id": page.pageid if hasattr(page, "pageid") else hash(title),
        }
    except wikipedia.exceptions.DisambiguationError as e:
        if e.options and len(e.options) > 0:
            return _fetch_single_page(e.options[0])  # Try first option
    except (wikipedia.exceptions.PageError, wikipedia.exceptions.RedirectError):
        pass
    except Exception as e:
        # Log but don't crash
        print(f"  Unexpected error fetching '{title}': {type(e).__name__}")

    return None


def _get_first_paragraphs(content: str, num_paras: int = 2) -> List[str]:
    """Extract first N substantive paragraphs from Wikipedia content."""
    paragraphs = content.split("\n\n")
    relevant = []

    for para in paragraphs:
        para = para.strip()
        # Skip headers, empty, very short paragraphs
        if not para or para.startswith("==") or len(para) < 80:
            continue
        relevant.append(para[:400])  # Limit each paragraph to 400 chars
        if len(relevant) >= num_paras:
            break

    return relevant


def extract_all_chunks(
    claims: List[str], claim_articles: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract text chunks for all claims.
    Returns dict: claim → list of chunk dicts
    """
    claim_chunks = {}

    for claim in claims:
        chunks = []
        articles = claim_articles.get(claim, [])

        for article in articles:
            if not article:
                continue

            # Add summary chunk
            summary = article.get("summary", "")
            if summary and len(summary) > 50:
                chunks.append(
                    {
                        "title": article["title"],
                        "url": article.get("url", ""),
                        "content": summary[:400],
                        "source_type": "summary",
                    }
                )

            # Add paragraph chunks
            for para in article.get("content_first_para", []):
                if para and len(para) > 50:
                    chunks.append(
                        {
                            "title": article["title"],
                            "url": article.get("url", ""),
                            "content": para[:400],
                            "source_type": "paragraph",
                        }
                    )

        claim_chunks[claim] = chunks

    return claim_chunks


def batch_score_chunks(
    claims: List[str], claim_chunks: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Score chunks by relevance using embeddings.
    Batches all embeddings together for efficiency.
    """
    # Collect all texts that need embedding
    all_texts = []
    text_mapping = []  # Track which claim each text belongs to

    for claim in claims:
        all_texts.append(claim)  # Claim itself
        text_mapping.append((claim, "claim", None))

        for chunk in claim_chunks.get(claim, []):
            all_texts.append(chunk["content"][:300])  # Only embed first 300 chars
            text_mapping.append((claim, "chunk", chunk))

    # Batch embed all texts at once
    if not all_texts:
        return {}

    try:
        all_embeddings = embeddings.embed_documents(all_texts)
    except Exception as e:
        print(f"  Batch embedding failed: {e}")
        # Fall back to individual embedding
        return _individual_score_chunks(claims, claim_chunks)

    # Organize embeddings
    claim_embeddings = {}
    chunk_embeddings = {}

    for i, (claim, text_type, chunk) in enumerate(text_mapping):
        if text_type == "claim":
            claim_embeddings[claim] = np.array(all_embeddings[i])
        else:
            if claim not in chunk_embeddings:
                chunk_embeddings[claim] = []
            chunk_embeddings[claim].append((chunk, np.array(all_embeddings[i])))

    # Score chunks
    evidence = {}

    for claim in claims:
        claim_emb = claim_embeddings.get(claim)
        chunks = chunk_embeddings.get(claim, [])

        if claim_emb is None or not chunks:
            evidence[claim] = [
                {
                    "title": "No evidence",
                    "content": f"No Wikipedia evidence found for: {claim[:100]}",
                    "url": "",
                    "relevance": 0.0,
                }
            ]
            continue

        scored = []
        for chunk, chunk_emb in chunks:
            # Cosine similarity
            similarity = np.dot(claim_emb, chunk_emb) / (
                np.linalg.norm(claim_emb) * np.linalg.norm(chunk_emb) + 1e-8
            )

            scored.append(
                {
                    "title": chunk["title"],
                    "url": chunk.get("url", ""),
                    "content": chunk["content"],
                    "relevance": round(float(similarity), 4),
                    "source_type": chunk.get("source_type", ""),
                }
            )

        # Sort and keep top 2
        scored.sort(key=lambda x: x["relevance"], reverse=True)
        evidence[claim] = scored[:2]

    return evidence


def _individual_score_chunks(
    claims: List[str], claim_chunks: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Fallback: Score chunks individually if batch fails."""
    evidence = {}

    for claim in claims:
        try:
            claim_emb = np.array(embeddings.embed_query(claim))
            chunks = claim_chunks.get(claim, [])

            if not chunks:
                evidence[claim] = [
                    {
                        "title": "No evidence",
                        "content": "No evidence found.",
                        "url": "",
                        "relevance": 0.0,
                    }
                ]
                continue

            scored = []
            for chunk in chunks:
                chunk_emb = np.array(embeddings.embed_query(chunk["content"][:300]))
                similarity = np.dot(claim_emb, chunk_emb) / (
                    np.linalg.norm(claim_emb) * np.linalg.norm(chunk_emb) + 1e-8
                )
                scored.append({**chunk, "relevance": round(float(similarity), 4)})

            scored.sort(key=lambda x: x["relevance"], reverse=True)
            evidence[claim] = scored[:2]
        except Exception as e:
            print(f"  Scoring failed for claim: {e}")
            evidence[claim] = [
                {
                    "title": "Error",
                    "content": "Scoring failed.",
                    "url": "",
                    "relevance": 0.0,
                }
            ]

    return evidence


# Quick test
if __name__ == "__main__":
    test_state = {
        "claims": [
            "The Eiffel Tower was completed in 1889",
            "Python was created by Guido van Rossum in 1991",
        ]
    }

    result = retrieval_node(test_state)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for claim, evidence_list in result["evidence"].items():
        print(f"\nClaim: {claim}")
        for i, ev in enumerate(evidence_list, 1):
            print(f"  [{ev['relevance']:.2%}] {ev['title']}: {ev['content'][:100]}...")

# Single LLM Call: Generate queries for all claims in one LLM call instead of multiple calls, making it faster and cheaper.
# Article Caching: Store fetched articles so the same article isn't downloaded again.
# Batch Embeddings: Process many text chunks together in one embedding request instead of one by one.
# Reduced Content: Use shorter summaries, fewer paragraphs, shorter embedding text, and fewer articles to reduce processing time.
# Defensive Fixes: Add safe checks, error handling, and backup methods to prevent crashes.
# Smaller Context Window: Reduce context size from 4096 to 2048 since query generation doesn't need a large context, improving speed and memory usage.

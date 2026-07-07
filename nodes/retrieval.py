"""
retrieval.py - Evidence Retrieval Node

Optimized for GTX 1650 hardware:
- Single LLM call for all query generation
- Article caching across claims
- Batched embedding calls
- Deduplicated evidence per claim
- Defensive error handling throughout
"""

import json
import re
import time
from typing import Any, Dict, List

import numpy as np
import wikipedia
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama

# Initialize models once at module level
llm = ChatOllama(model="qwen2.5:7b", temperature=0, num_ctx=2048)
embeddings = OllamaEmbeddings(model="nomic-embed-text")


# Module-level cache to avoid re-fetching same articles
_article_cache: Dict[str, Dict[str, Any]] = {}


def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve Wikipedia evidence for all claims.

    Input:  state["claims"] - list of factual claims
    Output: state["evidence"] - dict mapping claim → list of evidence dicts

    Each evidence dict has: title, url, content, relevance, source_type
    """
    claims = state.get("claims", [])

    if not claims:
        return {"evidence": {}}

    start_time = time.time()

    # Step 1: Generate all queries in one LLM call
    print(f"\n🔍 Generating search queries for {len(claims)} claims...")
    all_queries = generate_all_queries(claims)

    # Step 2: Fetch all unique Wikipedia articles
    print("📚 Fetching Wikipedia articles...")
    claim_articles = collect_all_articles(all_queries)

    # Step 3: Extract text chunks from articles
    print("✂️  Extracting text chunks...")
    claim_chunks = extract_all_chunks(claims, claim_articles)

    # Step 4: Score chunks by relevance (batched)
    print("🧮 Computing relevance scores...")
    evidence = batch_score_chunks(claims, claim_chunks)

    elapsed = time.time() - start_time
    print(f"✅ Retrieval complete in {elapsed:.1f}s")

    return {"evidence": evidence}


def generate_all_queries(claims: List[str]) -> Dict[str, List[str]]:
    """
    Generate search queries for ALL claims in ONE LLM call.
    Biggest performance win - 1 call instead of N.
    """
    claims_text = "\n".join([f"{i + 1}. {claim}" for i, claim in enumerate(claims)])

    prompt = f"""For each claim below, generate 2 Wikipedia search queries to find evidence.

Claims:
{claims_text}

Return a JSON object:
{{
    "1": ["query1", "query2"],
    "2": ["query1", "query2"]
}}

Rules:
- Each query max 6 words
- Focus on key entities, proper nouns, dates, numbers
- No punctuation
- Return ONLY the JSON object"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Extract JSON object
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            queries_map = json.loads(json_match.group(0))

            # Map back to claims
            result = {}
            for i, claim in enumerate(claims):
                key = str(i + 1)
                if key in queries_map and isinstance(queries_map[key], list):
                    result[claim] = queries_map[key][:2]
                else:
                    result[claim] = [clean_query_fallback(claim)]
            return result

    except Exception as e:
        print(f"  Batch query generation failed: {e}")

    # Complete fallback
    return {claim: [clean_query_fallback(claim)] for claim in claims}


def clean_query_fallback(claim: str) -> str:
    """
    Extract key terms from claim for Wikipedia search.
    Filters out vague words that produce garbage queries.
    """
    # Extended stop words including vague terms
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
        "their",
        "them",
        "and",
        "or",
        "but",
        "if",
        "while",
        # Vague terms that produce bad queries
        "ability",
        "abilities",
        "integration",
        "enhances",
        "enables",
        "allows",
        "provides",
        "offers",
        "various",
        "across",
        "experience",
        "operational",
        "efficiency",
        "industries",
        "perform",
        "tasks",
        "real-time",
        "including",
        "such",
        "however",
        "therefore",
        "moreover",
        "furthermore",
        "significantly",
        "particularly",
        "typically",
        "generally",
    }

    # Remove punctuation and split
    words = re.sub(r"[^\w\s-]", " ", claim).split()

    # Filter out stop words and short words
    key_terms = [w for w in words if w.lower() not in stop_words and len(w) > 2]

    # Prioritize proper nouns and specific terms
    proper_nouns = [t for t in key_terms if t[0].isupper() and len(t) > 3]
    numbers = [t for t in key_terms if t.replace("-", "").replace(".", "").isdigit()]
    long_terms = [t for t in key_terms if len(t) > 6]

    # Build query from most specific terms first
    specific_terms = proper_nouns + numbers + long_terms

    if specific_terms:
        return " ".join(specific_terms[:4])
    elif key_terms:
        return " ".join(key_terms[:4])
    else:
        return "Wikipedia"


def collect_all_articles(
    all_queries: Dict[str, List[str]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch all unique Wikipedia articles needed.
    Uses module-level cache to avoid re-fetching.
    """
    # Collect all unique queries
    unique_queries = set()
    for queries in all_queries.values():
        unique_queries.update(queries)

    # Fetch articles for each unique query (cached)
    query_articles = {}
    for query in unique_queries:
        query_articles[query] = fetch_wikipedia_articles(query)

    # Map claims to their articles (deduplicated by page_id)
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

        claim_articles[claim] = articles_for_claim[:3]  # Max 3 per claim

    return claim_articles


def fetch_wikipedia_articles(query: str) -> List[Dict[str, Any]]:
    """
    Fetch Wikipedia articles for a search query.
    Results are cached by query string.
    """
    cache_key = query.lower().strip()

    if cache_key in _article_cache:
        return _article_cache[cache_key]

    articles = []

    try:
        results = wikipedia.search(query, results=2)

        for title in results:
            article = _fetch_single_page(title)
            if article:
                articles.append(article)

    except Exception as e:
        print(f"  Search failed for '{query}': {str(e)[:80]}")

    _article_cache[cache_key] = articles
    return articles


def _fetch_single_page(title: str) -> Dict[str, Any]:
    """
    Safely fetch a single Wikipedia page.
    Returns None on any error (no crashes).
    """
    # Check cache first
    if title in _article_cache:
        return _article_cache[title]

    try:
        page = wikipedia.page(title, auto_suggest=False)

        return {
            "title": page.title,
            "url": page.url,
            "summary": page.summary[:500],
            "paragraphs": _get_first_paragraphs(page.content, num_paras=2),
            "page_id": getattr(page, "pageid", hash(title)),
        }

    except wikipedia.exceptions.DisambiguationError as e:
        # Try first disambiguation option
        if e.options and len(e.options) > 0:
            return _fetch_single_page(e.options[0])

    except wikipedia.exceptions.PageError:
        pass

    except wikipedia.exceptions.RedirectError:
        pass

    except Exception as e:
        print(f"  Error fetching '{title}': {type(e).__name__}")

    return None


def _get_first_paragraphs(content: str, num_paras: int = 2) -> List[str]:
    """
    Extract first N substantive paragraphs from Wikipedia content.
    Skips headers, empty lines, and very short paragraphs.
    """
    paragraphs = content.split("\n\n")
    relevant = []

    for para in paragraphs:
        para = para.strip()
        # Skip headers (== Section ==), empty, or very short paragraphs
        if not para or para.startswith("==") or len(para) < 80:
            continue
        relevant.append(para[:400])  # Limit each to 400 chars
        if len(relevant) >= num_paras:
            break

    return relevant


def extract_all_chunks(
    claims: List[str], claim_articles: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract text chunks from articles for each claim.
    Returns dict: claim → list of chunk dicts
    """
    claim_chunks = {}

    for claim in claims:
        chunks = []
        articles = claim_articles.get(claim, [])

        for article in articles:
            if not article:
                continue

            title = article.get("title", "Unknown")
            url = article.get("url", "")

            # Add summary chunk
            summary = article.get("summary", "")
            if summary and len(summary) > 50:
                chunks.append(
                    {
                        "title": title,
                        "url": url,
                        "content": summary[:400],
                        "source_type": "summary",
                    }
                )

            # Add paragraph chunks
            for para in article.get("paragraphs", []):
                if para and len(para) > 50:
                    chunks.append(
                        {
                            "title": title,
                            "url": url,
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
    Score chunks by semantic similarity to claims.
    Batches all embeddings together for efficiency.
    Deduplicates by title so verify node sees unique sources.
    """
    # Collect all texts that need embedding
    all_texts = []
    text_mapping = []  # Track (claim, type, chunk) for each text

    for claim in claims:
        all_texts.append(claim)
        text_mapping.append((claim, "claim", None))

        for chunk in claim_chunks.get(claim, []):
            all_texts.append(chunk["content"][:300])
            text_mapping.append((claim, "chunk", chunk))

    if not all_texts:
        return {}

    # Batch embed everything
    try:
        all_embeddings = embeddings.embed_documents(all_texts)
    except Exception as e:
        print(f"  Batch embedding failed: {e}")
        return _individual_score_chunks(claims, claim_chunks)

    # Organize embeddings by claim
    claim_embeddings = {}
    chunk_embeddings = {}

    for i, (claim, text_type, chunk) in enumerate(text_mapping):
        emb = np.array(all_embeddings[i])

        if text_type == "claim":
            claim_embeddings[claim] = emb
        else:
            if claim not in chunk_embeddings:
                chunk_embeddings[claim] = []
            chunk_embeddings[claim].append((chunk, emb))

    # Score and deduplicate
    evidence = {}

    for claim in claims:
        claim_emb = claim_embeddings.get(claim)
        chunks = chunk_embeddings.get(claim, [])

        if claim_emb is None or not chunks:
            evidence[claim] = [
                {
                    "title": "No evidence found",
                    "content": f"No Wikipedia evidence for: {claim[:100]}",
                    "url": "",
                    "relevance": 0.0,
                    "source_type": "fallback",
                }
            ]
            continue

        # Calculate cosine similarity for each chunk
        scored = []
        for chunk, chunk_emb in chunks:
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

        # Sort by relevance (highest first)
        scored.sort(key=lambda x: x["relevance"], reverse=True)

        # Deduplicate by title (CRITICAL FIX)
        # Ensures verify node sees unique sources
        seen_titles = set()
        deduped = []
        for item in scored:
            if item["title"] not in seen_titles:
                seen_titles.add(item["title"])
                deduped.append(item)

        evidence[claim] = deduped[:2]  # Top 2 unique articles

    return evidence


def _individual_score_chunks(
    claims: List[str], claim_chunks: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fallback: Score chunks individually if batch embedding fails.
    Also deduplicates by title.
    """
    evidence = {}

    for claim in claims:
        try:
            claim_emb = np.array(embeddings.embed_query(claim))
            chunks = claim_chunks.get(claim, [])

            if not chunks:
                evidence[claim] = [
                    {
                        "title": "No evidence found",
                        "content": "No evidence found.",
                        "url": "",
                        "relevance": 0.0,
                        "source_type": "fallback",
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

            # Sort and deduplicate
            scored.sort(key=lambda x: x["relevance"], reverse=True)

            seen_titles = set()
            deduped = []
            for item in scored:
                if item["title"] not in seen_titles:
                    seen_titles.add(item["title"])
                    deduped.append(item)

            evidence[claim] = deduped[:2]

        except Exception as e:
            print(f"  Scoring failed for claim: {e}")
            evidence[claim] = [
                {
                    "title": "Error",
                    "content": "Scoring failed.",
                    "url": "",
                    "relevance": 0.0,
                    "source_type": "error",
                }
            ]

    return evidence


# Test it directly
if __name__ == "__main__":
    print("=" * 70)
    print("TESTING RETRIEVAL NODE")
    print("=" * 70)

    test_state = {
        "claims": [
            "The Eiffel Tower was completed in 1889",
            "Python was created by Guido van Rossum in 1991",
        ]
    }

    start = time.time()
    result = retrieval_node(test_state)
    elapsed = time.time() - start

    print(f"\n Total time taken: {elapsed:.2f}s")

    for claim, evidence_list in result["evidence"].items():
        print(f"\nClaim: {claim}")
        if not evidence_list:
            print(" No evidence found")
        for i, ev in enumerate(evidence_list, 1):
            relevance = ev.get("relevance", 0)
            print(f"   {i}. [{relevance:.1%}] {ev['title']}")
            print(f"      {ev['content'][:120]}...")

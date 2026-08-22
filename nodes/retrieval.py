"""
retrieval.py - Evidence Retrieval Node

Optimized for GTX 1650 hardware:
- Single LLM call for all query generation
- Multi-source evidence: Wikipedia (batch API) + arXiv (abstracts)
- Persistent disk cache per source (never re-fetches same article)
- Batched embedding calls
- Deduplicated evidence per claim, tagged by source
- Defensive error handling throughout
"""

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import numpy as np
import requests
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ---- Wikipedia API session ----
_wiki_session = requests.Session()
_wiki_session.headers.update(
    {
        "User-Agent": "HallucinationRadar/1.0 (research project; direct API)",
        "Accept": "application/json",
    }
)
WIKI_API = "https://en.wikipedia.org/w/api.php"

# ---- arXiv API session ----
_arxiv_session = requests.Session()
_arxiv_session.headers.update(
    {
        "User-Agent": "HallucinationRadar/1.0 (research project; direct API)",
    }
)
ARXIV_API = "http://export.arxiv.org/api/query"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
WIKI_CACHE_FILE = os.path.join(DATA_DIR, "wiki_cache.json")
ARXIV_CACHE_FILE = os.path.join(DATA_DIR, "arxiv_cache.json")

# Initialize models once at module level
llm = ChatOllama(model="qwen2.5:7b", temperature=0, num_ctx=2048)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# In-memory caches (loaded from disk at startup), one per source
_wiki_cache: Dict[str, Any] = {}
_arxiv_cache: Dict[str, Any] = {}


# =========================================================
# Cache load/save (separate files per source)
# =========================================================

def load_disk_caches() -> None:
    """Load both persistent caches from disk into memory."""
    global _wiki_cache, _arxiv_cache

    for path, target_name in [(WIKI_CACHE_FILE, "wiki"), (ARXIV_CACHE_FILE, "arxiv")]:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                if target_name == "wiki":
                    _wiki_cache = data
                else:
                    _arxiv_cache = data
                print(f"  Loaded {len(data)} cached {target_name} entries from disk")
        except Exception as e:
            print(f"  {target_name} cache load failed: {e}, starting fresh")


def save_disk_caches() -> None:
    """Save both in-memory caches to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for path, data in [(WIKI_CACHE_FILE, _wiki_cache), (ARXIV_CACHE_FILE, _arxiv_cache)]:
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"  Cache save failed for {path}: {e}")


# =========================================================
# Main entry point
# =========================================================

def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve evidence for all claims from Wikipedia AND arXiv.

    Input:  state["claims"] - list of factual claims
    Output: state["evidence"] - dict mapping claim -> list of evidence dicts,
            each tagged with a "source" field ("wikipedia" or "arxiv")
    """
    claims = state.get("claims", [])

    if not claims:
        return {"evidence": {}}

    load_disk_caches()

    start_time = time.time()

    print(f"\nGenerating search queries for {len(claims)} claims...")
    all_queries = generate_all_queries(claims)

    print("Fetching Wikipedia articles...")
    claim_wiki_articles = collect_all_wiki_articles(all_queries)

    print("Fetching arXiv papers...")
    claim_arxiv_articles = collect_all_arxiv_papers(all_queries)

    print("Extracting text chunks...")
    claim_chunks = extract_all_chunks(claims, claim_wiki_articles, claim_arxiv_articles)

    print("Computing relevance scores...")
    evidence = batch_score_chunks(claims, claim_chunks)

    save_disk_caches()

    elapsed = time.time() - start_time
    print(f"Retrieval complete in {elapsed:.1f}s")

    return {"evidence": evidence}


# =========================================================
# Query generation (shared across both sources)
# =========================================================

def generate_all_queries(claims: List[str]) -> Dict[str, List[str]]:
    """
    Generate search queries for ALL claims in ONE LLM call.
    Same queries are reused for both Wikipedia and arXiv search.
    """
    claims_text = "\n".join([f"{i + 1}. {claim}" for i, claim in enumerate(claims)])

    prompt = f"""For each claim below, generate 2 search queries to find evidence.
These queries will be used against both Wikipedia and arXiv (research papers).

Claims:
{claims_text}

Return a JSON object:
{{
    "1": ["query1", "query2"],
    "2": ["query1", "query2"]
}}

Rules:
- Each query max 6 words
- Search for the main named entity only (e.g. "GPT-3" not "GPT-3 parameters training")
- Use exact names a search engine would recognize
- One concept per query
- No punctuation
- Return ONLY the JSON object"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            queries_map = json.loads(json_match.group(0))

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

    return {claim: [clean_query_fallback(claim)] for claim in claims}


def clean_query_fallback(claim: str) -> str:
    """Extract key terms from claim for search fallback."""
    stop_words = {
        "the", "a", "an", "is", "was", "were", "are", "been",
        "has", "have", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "shall", "to",
        "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "this", "that", "these", "those", "it", "its", "they",
        "their", "them", "and", "or", "but", "if", "while",
        "ability", "abilities", "integration", "enhances", "enables",
        "allows", "provides", "offers", "various", "across", "experience",
        "operational", "efficiency", "industries", "perform", "tasks",
        "real-time", "including", "such", "however", "therefore",
        "moreover", "furthermore", "significantly", "particularly",
        "typically", "generally",
    }

    words = re.sub(r"[^\w\s-]", " ", claim).split()
    key_terms = [w for w in words if w.lower() not in stop_words and len(w) > 2]

    proper_nouns = [t for t in key_terms if t[0].isupper() and len(t) > 3]
    numbers = [t for t in key_terms if t.replace("-", "").replace(".", "").isdigit()]
    long_terms = [t for t in key_terms if len(t) > 6]

    specific_terms = proper_nouns + numbers + long_terms

    if specific_terms:
        return " ".join(specific_terms[:4])
    elif key_terms:
        return " ".join(key_terms[:4])
    else:
        return "large language model"


# =========================================================
# Wikipedia source
# =========================================================

def wiki_search(query: str, num_results: int = 2) -> List[str]:
    """Search Wikipedia for page titles matching a query."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": num_results,
        "format": "json",
    }

    for attempt in range(4):
        try:
            time.sleep(0.5 * (attempt + 1))
            r = _wiki_session.get(WIKI_API, params=params, timeout=10)

            if r.status_code == 429:
                wait = (attempt + 1) * 5
                print(f"  429 on wiki search '{query}', waiting {wait}s (attempt {attempt + 1}/4)...")
                time.sleep(wait)
                continue

            r.raise_for_status()
            data = r.json()
            results = data.get("query", {}).get("search", [])
            return [item["title"] for item in results]

        except Exception as e:
            if "429" in str(e):
                wait = (attempt + 1) * 5
                print(f"  429 on wiki search '{query}', waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Wiki search failed '{query}': {type(e).__name__}: {str(e)[:60]}")
                return []

    print(f"  Wiki search gave up: '{query}'")
    return []


def wiki_batch_fetch(titles: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch multiple Wikipedia pages in ONE API request (up to 50 per request)."""
    if not titles:
        return {}

    titles_str = "|".join(titles[:50])

    params = {
        "action": "query",
        "prop": "extracts|info",
        "exintro": False,
        "explaintext": True,
        "inprop": "url",
        "redirects": 1,
        "titles": titles_str,
        "format": "json",
    }

    for attempt in range(4):
        try:
            time.sleep(0.5 * (attempt + 1))
            r = _wiki_session.get(WIKI_API, params=params, timeout=30)

            if r.status_code == 429:
                wait = (attempt + 1) * 5
                print(f"  429 on wiki batch fetch, waiting {wait}s (attempt {attempt + 1}/4)...")
                time.sleep(wait)
                continue

            r.raise_for_status()
            data = r.json()

            pages = data.get("query", {}).get("pages", {})
            result = {}

            for page in pages.values():
                if page.get("pageid", -1) == -1:
                    continue

                content = page.get("extract", "")
                if not content:
                    continue

                title = page.get("title", "Unknown")
                result[title] = {
                    "title": title,
                    "url": page.get("fullurl", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"),
                    "summary": content[:500],
                    "paragraphs": _get_first_paragraphs(content, num_paras=2),
                    "page_id": page.get("pageid", hash(title)),
                    "source": "wikipedia",
                }

            return result

        except Exception as e:
            if "429" in str(e):
                wait = (attempt + 1) * 5
                print(f"  429 on wiki batch fetch, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Wiki batch fetch failed: {type(e).__name__}: {str(e)[:60]}")
                return {}

    print("  Wiki batch fetch gave up after 4 attempts")
    return {}


def collect_all_wiki_articles(
    all_queries: Dict[str, List[str]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch all unique Wikipedia articles needed, using cache + batch fetching."""
    unique_queries = set()
    for queries in all_queries.values():
        unique_queries.update(queries)

    query_to_titles: Dict[str, List[str]] = {}
    search_cache_key = lambda q: f"search:{q.lower().strip()}"

    uncached_queries = []
    for query in unique_queries:
        key = search_cache_key(query)
        if key in _wiki_cache:
            query_to_titles[query] = _wiki_cache[key]
        else:
            uncached_queries.append(query)

    for query in uncached_queries:
        titles = wiki_search(query, num_results=2)
        query_to_titles[query] = titles
        _wiki_cache[search_cache_key(query)] = titles

    all_titles_needed = set()
    for titles in query_to_titles.values():
        all_titles_needed.update(titles)

    uncached_titles = [t for t in all_titles_needed if t not in _wiki_cache]

    if uncached_titles:
        print(f"  Batch fetching {len(uncached_titles)} wiki articles in "
              f"{(len(uncached_titles) + 49) // 50} request(s)...")

        for i in range(0, len(uncached_titles), 50):
            batch = uncached_titles[i:i + 50]
            fetched = wiki_batch_fetch(batch)

            for title, article in fetched.items():
                _wiki_cache[title] = article

            for title in batch:
                if title not in _wiki_cache:
                    _wiki_cache[title] = None
    else:
        print("  All Wikipedia articles loaded from cache!")

    claim_articles = {}
    for claim, queries in all_queries.items():
        articles_for_claim = []
        seen_ids = set()

        for query in queries:
            for title in query_to_titles.get(query, []):
                article = _wiki_cache.get(title)
                if not article:
                    continue
                page_id = article.get("page_id")
                if page_id and page_id not in seen_ids:
                    seen_ids.add(page_id)
                    articles_for_claim.append(article)

        claim_articles[claim] = articles_for_claim[:3]

    return claim_articles


def _get_first_paragraphs(content: str, num_paras: int = 2) -> List[str]:
    """Extract first N substantive paragraphs from article content."""
    paragraphs = content.split("\n\n")
    relevant = []

    for para in paragraphs:
        para = para.strip()
        if not para or para.startswith("==") or len(para) < 80:
            continue
        relevant.append(para[:400])
        if len(relevant) >= num_paras:
            break

    return relevant


# =========================================================
# arXiv source
# =========================================================

_ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def arxiv_search(query: str, num_results: int = 2) -> List[Dict[str, Any]]:
    """
    Search arXiv for papers matching a query.
    Returns list of paper dicts with title, summary (abstract), url, id.
    No API key needed. Free public API.
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": num_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    for attempt in range(3):
        try:
            time.sleep(0.5 * (attempt + 1))  # arXiv asks for max 1 req / 3s, being conservative
            r = _arxiv_session.get(ARXIV_API, params=params, timeout=15)
            r.raise_for_status()

            root = ET.fromstring(r.text)
            entries = root.findall("atom:entry", _ARXIV_NS)

            papers = []
            for entry in entries:
                arxiv_id_full = entry.findtext("atom:id", default="", namespaces=_ARXIV_NS)
                arxiv_id = arxiv_id_full.rsplit("/", 1)[-1] if arxiv_id_full else ""

                title = entry.findtext("atom:title", default="", namespaces=_ARXIV_NS)
                title = re.sub(r"\s+", " ", title).strip()

                summary = entry.findtext("atom:summary", default="", namespaces=_ARXIV_NS)
                summary = re.sub(r"\s+", " ", summary).strip()

                published = entry.findtext("atom:published", default="", namespaces=_ARXIV_NS)
                year = published[:4] if published else ""

                if not title or not summary:
                    continue

                papers.append({
                    "title": f"{title} (arXiv{', ' + year if year else ''})",
                    "url": arxiv_id_full or f"https://arxiv.org/abs/{arxiv_id}",
                    "summary": summary[:500],
                    "paragraphs": [summary[:800]],  # abstract is the whole "paragraph"
                    "page_id": arxiv_id or hash(title),
                    "source": "arxiv",
                })

            return papers

        except ET.ParseError as e:
            print(f"  arXiv response parse failed for '{query}': {e}")
            return []
        except Exception as e:
            if attempt < 2:
                wait = (attempt + 1) * 3
                print(f"  arXiv search error on '{query}', retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  arXiv search failed '{query}': {type(e).__name__}: {str(e)[:60]}")
                return []

    return []


def collect_all_arxiv_papers(
    all_queries: Dict[str, List[str]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch arXiv papers for all claims, using cache to avoid repeat queries."""
    unique_queries = set()
    for queries in all_queries.values():
        unique_queries.update(queries)

    query_to_papers: Dict[str, List[Dict[str, Any]]] = {}
    search_cache_key = lambda q: f"search:{q.lower().strip()}"

    fetched_count = 0
    for query in unique_queries:
        key = search_cache_key(query)
        if key in _arxiv_cache:
            query_to_papers[query] = _arxiv_cache[key]
        else:
            papers = arxiv_search(query, num_results=2)
            query_to_papers[query] = papers
            _arxiv_cache[key] = papers
            fetched_count += 1

    if fetched_count == 0:
        print("  All arXiv results loaded from cache!")
    else:
        print(f"  Fetched arXiv results for {fetched_count} new quer{'y' if fetched_count == 1 else 'ies'}")

    claim_papers = {}
    for claim, queries in all_queries.items():
        papers_for_claim = []
        seen_ids = set()

        for query in queries:
            for paper in query_to_papers.get(query, []):
                paper_id = paper.get("page_id")
                if paper_id and paper_id not in seen_ids:
                    seen_ids.add(paper_id)
                    papers_for_claim.append(paper)

        claim_papers[claim] = papers_for_claim[:2]  # cap arXiv results per claim

    return claim_papers


# =========================================================
# Chunking (shared across both sources)
# =========================================================

def extract_all_chunks(
    claims: List[str],
    claim_wiki_articles: Dict[str, List[Dict[str, Any]]],
    claim_arxiv_articles: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract text chunks from both Wikipedia and arXiv sources for each claim."""
    claim_chunks = {}

    for claim in claims:
        chunks = []

        for article in claim_wiki_articles.get(claim, []) + claim_arxiv_articles.get(claim, []):
            if not article:
                continue

            title = article.get("title", "Unknown")
            url = article.get("url", "")
            source = article.get("source", "wikipedia")

            summary = article.get("summary", "")
            if summary and len(summary) > 50:
                chunks.append({
                    "title": title,
                    "url": url,
                    "content": summary[:400],
                    "source_type": "summary",
                    "source": source,
                })

            for para in article.get("paragraphs", []):
                if para and len(para) > 50:
                    chunks.append({
                        "title": title,
                        "url": url,
                        "content": para[:400],
                        "source_type": "paragraph",
                        "source": source,
                    })

        claim_chunks[claim] = chunks

    return claim_chunks


# =========================================================
# Scoring (shared across both sources)
# =========================================================

def batch_score_chunks(
    claims: List[str], claim_chunks: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Score chunks by semantic similarity to claims.
    Batches all embeddings together for efficiency.
    Deduplicates by title. Keeps top evidence regardless of source,
    so Wikipedia and arXiv compete on relevance alone.
    """
    all_texts = []
    text_mapping = []

    for claim in claims:
        all_texts.append(claim)
        text_mapping.append((claim, "claim", None))

        for chunk in claim_chunks.get(claim, []):
            all_texts.append(chunk["content"][:300])
            text_mapping.append((claim, "chunk", chunk))

    if not all_texts:
        return {}

    try:
        all_embeddings = embeddings.embed_documents(all_texts)
    except Exception as e:
        print(f"  Batch embedding failed: {e}")
        return _individual_score_chunks(claims, claim_chunks)

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

    evidence = {}

    for claim in claims:
        claim_emb = claim_embeddings.get(claim)
        chunks = chunk_embeddings.get(claim, [])

        if claim_emb is None or not chunks:
            evidence[claim] = [{
                "title": "No evidence found",
                "content": f"No evidence found for: {claim[:100]}",
                "url": "",
                "relevance": 0.0,
                "source_type": "fallback",
                "source": "none",
            }]
            continue

        scored = []
        for chunk, chunk_emb in chunks:
            similarity = np.dot(claim_emb, chunk_emb) / (
                np.linalg.norm(claim_emb) * np.linalg.norm(chunk_emb) + 1e-8
            )
            scored.append({
                "title": chunk["title"],
                "url": chunk.get("url", ""),
                "content": chunk["content"],
                "relevance": round(float(similarity), 4),
                "source_type": chunk.get("source_type", ""),
                "source": chunk.get("source", "wikipedia"),
            })

        scored.sort(key=lambda x: x["relevance"], reverse=True)

        seen_titles = set()
        deduped = []
        for item in scored:
            if item["title"] not in seen_titles:
                seen_titles.add(item["title"])
                deduped.append(item)

        # Keep top 3 now instead of 2, since we may have two sources competing
        evidence[claim] = deduped[:3]

    return evidence


def _individual_score_chunks(
    claims: List[str], claim_chunks: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Fallback: score chunks individually if batch embedding fails."""
    evidence = {}

    for claim in claims:
        try:
            claim_emb = np.array(embeddings.embed_query(claim))
            chunks = claim_chunks.get(claim, [])

            if not chunks:
                evidence[claim] = [{
                    "title": "No evidence found",
                    "content": "No evidence found.",
                    "url": "",
                    "relevance": 0.0,
                    "source_type": "fallback",
                    "source": "none",
                }]
                continue

            scored = []
            for chunk in chunks:
                chunk_emb = np.array(embeddings.embed_query(chunk["content"][:300]))
                similarity = np.dot(claim_emb, chunk_emb) / (
                    np.linalg.norm(claim_emb) * np.linalg.norm(chunk_emb) + 1e-8
                )
                scored.append({**chunk, "relevance": round(float(similarity), 4)})

            scored.sort(key=lambda x: x["relevance"], reverse=True)

            seen_titles = set()
            deduped = []
            for item in scored:
                if item["title"] not in seen_titles:
                    seen_titles.add(item["title"])
                    deduped.append(item)

            evidence[claim] = deduped[:3]

        except Exception as e:
            print(f"  Scoring failed for claim: {e}")
            evidence[claim] = [{
                "title": "Error",
                "content": "Scoring failed.",
                "url": "",
                "relevance": 0.0,
                "source_type": "error",
                "source": "none",
            }]

    return evidence


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING MULTI-SOURCE RETRIEVAL NODE")
    print("=" * 70)

    test_state = {
        "claims": [
            "The Eiffel Tower was completed in 1889",
            "GPT-3 was developed by OpenAI",
            "The transformer architecture uses self-attention mechanisms",
        ]
    }

    start = time.time()
    result = retrieval_node(test_state)
    elapsed = time.time() - start

    print(f"\nTotal time: {elapsed:.2f}s")

    for claim, evidence_list in result["evidence"].items():
        print(f"\nClaim: {claim}")
        if not evidence_list:
            print("  No evidence found")
        for i, ev in enumerate(evidence_list, 1):
            relevance = ev.get("relevance", 0)
            source = ev.get("source", "?")
            print(f"  {i}. [{relevance:.1%}] ({source}) {ev['title']}")
            print(f"     {ev['content'][:120]}...")

# """
# retrieval.py - Evidence Retrieval Node

# Optimized for GTX 1650 hardware:
# - Single LLM call for all query generation
# - Direct Wikipedia API via requests (replaces wikipedia library)
# - Article caching across claims
# - Batched embedding calls
# - Deduplicated evidence per claim
# - Defensive error handling throughout
# """

# import json
# import re
# import time
# from typing import Any, Dict, List

# import numpy as np
# import requests
# from langchain_ollama import ChatOllama, OllamaEmbeddings

# # Wikipedia API session with proper headers
# _session = requests.Session()
# _session.headers.update(
#     {
#         "User-Agent": "HallucinationRadar/1.0 (research project; using direct API)",
#         "Accept": "application/json",
#     }
# )

# WIKI_API = "https://en.wikipedia.org/w/api.php"

# # Initialize models once at module level
# llm = ChatOllama(model="qwen2.5:7b", temperature=0, num_ctx=2048)
# embeddings = OllamaEmbeddings(model="nomic-embed-text")

# # Module-level cache to avoid re-fetching same articles
# _article_cache: Dict[str, Dict[str, Any]] = {}


# def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Retrieve Wikipedia evidence for all claims.

#     Input:  state["claims"] - list of factual claims
#     Output: state["evidence"] - dict mapping claim -> list of evidence dicts

#     Each evidence dict has: title, url, content, relevance, source_type
#     """
#     claims = state.get("claims", [])

#     if not claims:
#         return {"evidence": {}}

#     start_time = time.time()

#     print(f"\nGenerating search queries for {len(claims)} claims...")
#     all_queries = generate_all_queries(claims)

#     print("Fetching Wikipedia articles...")
#     claim_articles = collect_all_articles(all_queries)

#     print("Extracting text chunks...")
#     claim_chunks = extract_all_chunks(claims, claim_articles)

#     print("Computing relevance scores...")
#     evidence = batch_score_chunks(claims, claim_chunks)

#     elapsed = time.time() - start_time
#     print(f"Retrieval complete in {elapsed:.1f}s")

#     return {"evidence": evidence}


# def generate_all_queries(claims: List[str]) -> Dict[str, List[str]]:
#     """
#     Generate search queries for ALL claims in ONE LLM call.
#     Biggest performance win - 1 call instead of N.
#     """
#     claims_text = "\n".join([f"{i + 1}. {claim}" for i, claim in enumerate(claims)])

#     prompt = f"""For each claim below, generate 2 Wikipedia search queries to find evidence.

# Claims:
# {claims_text}

# Return a JSON object:
# {{
#     "1": ["query1", "query2"],
#     "2": ["query1", "query2"]
# }}

# Rules:
# - Each query max 6 words
# - Focus on key entities, proper nouns, dates, numbers
# - No punctuation
# - Return ONLY the JSON object"""

#     try:
#         response = llm.invoke(prompt)
#         content = response.content.strip()

#         json_match = re.search(r"\{.*\}", content, re.DOTALL)
#         if json_match:
#             queries_map = json.loads(json_match.group(0))

#             result = {}
#             for i, claim in enumerate(claims):
#                 key = str(i + 1)
#                 if key in queries_map and isinstance(queries_map[key], list):
#                     result[claim] = queries_map[key][:2]
#                 else:
#                     result[claim] = [clean_query_fallback(claim)]
#             return result

#     except Exception as e:
#         print(f"  Batch query generation failed: {e}")

#     return {claim: [clean_query_fallback(claim)] for claim in claims}


# def clean_query_fallback(claim: str) -> str:
#     """
#     Extract key terms from claim for Wikipedia search.
#     Filters out vague words that produce garbage queries.
#     """
#     stop_words = {
#         "the",
#         "a",
#         "an",
#         "is",
#         "was",
#         "were",
#         "are",
#         "been",
#         "has",
#         "have",
#         "had",
#         "do",
#         "does",
#         "did",
#         "will",
#         "would",
#         "could",
#         "should",
#         "may",
#         "might",
#         "can",
#         "shall",
#         "to",
#         "of",
#         "in",
#         "for",
#         "on",
#         "with",
#         "at",
#         "by",
#         "from",
#         "as",
#         "this",
#         "that",
#         "these",
#         "those",
#         "it",
#         "its",
#         "they",
#         "their",
#         "them",
#         "and",
#         "or",
#         "but",
#         "if",
#         "while",
#         "ability",
#         "abilities",
#         "integration",
#         "enhances",
#         "enables",
#         "allows",
#         "provides",
#         "offers",
#         "various",
#         "across",
#         "experience",
#         "operational",
#         "efficiency",
#         "industries",
#         "perform",
#         "tasks",
#         "real-time",
#         "including",
#         "such",
#         "however",
#         "therefore",
#         "moreover",
#         "furthermore",
#         "significantly",
#         "particularly",
#         "typically",
#         "generally",
#     }

#     words = re.sub(r"[^\w\s-]", " ", claim).split()
#     key_terms = [w for w in words if w.lower() not in stop_words and len(w) > 2]

#     proper_nouns = [t for t in key_terms if t[0].isupper() and len(t) > 3]
#     numbers = [t for t in key_terms if t.replace("-", "").replace(".", "").isdigit()]
#     long_terms = [t for t in key_terms if len(t) > 6]

#     specific_terms = proper_nouns + numbers + long_terms

#     if specific_terms:
#         return " ".join(specific_terms[:4])
#     elif key_terms:
#         return " ".join(key_terms[:4])
#     else:
#         return "large language model"


# def wiki_search(query: str, num_results: int = 2) -> List[str]:
#     """
#     Search Wikipedia for page titles matching a query.
#     Uses direct API with retry + exponential backoff.
#     """
#     params = {
#         "action": "query",
#         "list": "search",
#         "srsearch": query,
#         "srlimit": num_results,
#         "format": "json",
#     }

#     for attempt in range(4):
#         try:
#             time.sleep(1.0 + attempt)  # 1s, 2s, 3s, 4s
#             r = _session.get(WIKI_API, params=params, timeout=10)

#             if r.status_code == 429:
#                 wait = (attempt + 1) * 3  # 3s, 6s, 9s, 12s
#                 print(
#                     f"  429 on search '{query}', waiting {wait}s (attempt {attempt + 1}/4)..."
#                 )
#                 time.sleep(wait)
#                 continue

#             r.raise_for_status()
#             data = r.json()
#             results = data.get("query", {}).get("search", [])
#             return [item["title"] for item in results]

#         except Exception as e:
#             if "429" in str(e):
#                 wait = (attempt + 1) * 3
#                 print(
#                     f"  429 on search '{query}', waiting {wait}s (attempt {attempt + 1}/4)..."
#                 )
#                 time.sleep(wait)
#             else:
#                 print(
#                     f"  Search failed for '{query}': {type(e).__name__}: {str(e)[:60]}"
#                 )
#                 return []

#     print(f"  Search gave up after 4 attempts: '{query}'")
#     return []


# def wiki_fetch_page(title: str) -> Dict[str, Any]:
#     """
#     Fetch a Wikipedia page's extract and metadata.
#     Uses direct API with retry + exponential backoff.
#     """
#     params = {
#         "action": "query",
#         "prop": "extracts|info",
#         "exintro": False,
#         "explaintext": True,
#         "inprop": "url",
#         "redirects": 1,
#         "titles": title,
#         "format": "json",
#     }

#     for attempt in range(4):
#         try:
#             time.sleep(1.0 + attempt)  # 1s, 2s, 3s, 4s
#             r = _session.get(WIKI_API, params=params, timeout=10)

#             if r.status_code == 429:
#                 wait = (attempt + 1) * 3  # 3s, 6s, 9s, 12s
#                 print(
#                     f"  429 fetching '{title}', waiting {wait}s (attempt {attempt + 1}/4)..."
#                 )
#                 time.sleep(wait)
#                 continue

#             r.raise_for_status()
#             data = r.json()

#             pages = data.get("query", {}).get("pages", {})
#             page = next(iter(pages.values()))

#             if page.get("pageid", -1) == -1:
#                 return None

#             content = page.get("extract", "")
#             if not content:
#                 return None

#             return {
#                 "title": page.get("title", title),
#                 "url": page.get(
#                     "fullurl",
#                     f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
#                 ),
#                 "summary": content[:500],
#                 "paragraphs": _get_first_paragraphs(content, num_paras=2),
#                 "page_id": page.get("pageid", hash(title)),
#             }

#         except Exception as e:
#             if "429" in str(e):
#                 wait = (attempt + 1) * 3
#                 print(
#                     f"  429 fetching '{title}', waiting {wait}s (attempt {attempt + 1}/4)..."
#                 )
#                 time.sleep(wait)
#             else:
#                 print(f"  Error fetching '{title}': {type(e).__name__}: {str(e)[:60]}")
#                 return None

#     print(f"  Fetch gave up after 4 attempts: '{title}'")
#     return None


# def wiki_fetch_pages_batch(titles: List[str]) -> Dict[str, Dict[str, Any]]:
#     """
#     Fetch multiple Wikipedia pages in ONE API call.
#     Dramatically reduces 429 errors.
#     """
#     if not titles:
#         return {}

#     # Deduplicate titles
#     unique_titles = list(set(titles))
#     results = {}

#     # Process in batches of 10 (Wikipedia limit is 50, but 10 is safer)
#     batch_size = 10
#     for i in range(0, len(unique_titles), batch_size):
#         batch = unique_titles[i : i + batch_size]
#         titles_param = "|".join(batch)

#         params = {
#             "action": "query",
#             "prop": "extracts|info",
#             "exintro": False,
#             "explaintext": True,
#             "inprop": "url",
#             "redirects": 1,
#             "titles": titles_param,
#             "format": "json",
#         }

#         for attempt in range(4):
#             try:
#                 time.sleep(1.0 + attempt)
#                 r = _session.get(WIKI_API, params=params, timeout=15)

#                 if r.status_code == 429:
#                     wait = (attempt + 1) * 3
#                     print(
#                         f"  429 on batch fetch, waiting {wait}s (attempt {attempt + 1}/4)..."
#                     )
#                     time.sleep(wait)
#                     continue

#                 r.raise_for_status()
#                 data = r.json()

#                 pages = data.get("query", {}).get("pages", {})
#                 for page_id, page in pages.items():
#                     if page.get("pageid", -1) == -1:
#                         continue

#                     content = page.get("extract", "")
#                     if not content:
#                         continue

#                     title = page.get("title", "")
#                     results[title] = {
#                         "title": title,
#                         "url": page.get(
#                             "fullurl",
#                             f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
#                         ),
#                         "summary": content[:500],
#                         "paragraphs": _get_first_paragraphs(content, num_paras=2),
#                         "page_id": page.get("pageid", hash(title)),
#                     }

#                 break  # Success

#             except Exception as e:
#                 if attempt == 3:
#                     print(f"  Batch fetch failed: {type(e).__name__}: {str(e)[:60]}")

#         # Small delay between batches
#         if i + batch_size < len(unique_titles):
#             time.sleep(1)

#     return results


# def collect_all_articles(
#     all_queries: Dict[str, List[str]],
# ) -> Dict[str, List[Dict[str, Any]]]:
#     """
#     Fetch all unique Wikipedia articles needed.
#     Uses batch fetching to minimize API calls.
#     """
#     # Collect all unique queries
#     unique_queries = set()
#     for queries in all_queries.values():
#         unique_queries.update(queries)

#     # Step 1: Get article titles via search (still individual, but cached)
#     all_titles = set()
#     query_titles = {}

#     for query in unique_queries:
#         titles = wiki_search(query, num_results=2)
#         query_titles[query] = titles
#         all_titles.update(titles)

#     # Step 2: Fetch ALL pages in ONE batch call
#     print(f"  Batch fetching {len(all_titles)} pages...")
#     all_articles = wiki_fetch_pages_batch(list(all_titles))

#     # Step 3: Map queries to articles
#     query_articles = {}
#     for query, titles in query_titles.items():
#         articles = []
#         for title in titles:
#             if title in all_articles:
#                 articles.append(all_articles[title])
#         query_articles[query] = articles

#     # Step 4: Map claims to articles
#     claim_articles = {}
#     for claim, queries in all_queries.items():
#         articles_for_claim = []
#         seen_ids = set()

#         for query in queries:
#             for article in query_articles.get(query, []):
#                 page_id = article.get("page_id")
#                 if page_id and page_id not in seen_ids:
#                     seen_ids.add(page_id)
#                     articles_for_claim.append(article)

#         claim_articles[claim] = articles_for_claim[:3]

#     return claim_articles


# def fetch_wikipedia_articles(query: str) -> List[Dict[str, Any]]:
#     """
#     Search Wikipedia and fetch matching articles.
#     Cached by query string.
#     """
#     cache_key = query.lower().strip()

#     if cache_key in _article_cache:
#         return _article_cache[cache_key]

#     articles = []

#     titles = wiki_search(query, num_results=2)

#     for title in titles:
#         if title in _article_cache:
#             article = _article_cache[title]
#         else:
#             article = wiki_fetch_page(title)
#             if article:
#                 _article_cache[title] = article

#         if article:
#             articles.append(article)

#     _article_cache[cache_key] = articles
#     return articles


# def _get_first_paragraphs(content: str, num_paras: int = 2) -> List[str]:
#     """
#     Extract first N substantive paragraphs from Wikipedia content.
#     Skips headers, empty lines, and very short paragraphs.
#     """
#     paragraphs = content.split("\n\n")
#     relevant = []

#     for para in paragraphs:
#         para = para.strip()
#         if not para or para.startswith("==") or len(para) < 80:
#             continue
#         relevant.append(para[:400])
#         if len(relevant) >= num_paras:
#             break

#     return relevant


# def extract_all_chunks(
#     claims: List[str], claim_articles: Dict[str, List[Dict[str, Any]]]
# ) -> Dict[str, List[Dict[str, Any]]]:
#     """
#     Extract text chunks from articles for each claim.
#     Returns dict: claim -> list of chunk dicts
#     """
#     claim_chunks = {}

#     for claim in claims:
#         chunks = []
#         articles = claim_articles.get(claim, [])

#         for article in articles:
#             if not article:
#                 continue

#             title = article.get("title", "Unknown")
#             url = article.get("url", "")

#             summary = article.get("summary", "")
#             if summary and len(summary) > 50:
#                 chunks.append(
#                     {
#                         "title": title,
#                         "url": url,
#                         "content": summary[:400],
#                         "source_type": "summary",
#                     }
#                 )

#             for para in article.get("paragraphs", []):
#                 if para and len(para) > 50:
#                     chunks.append(
#                         {
#                             "title": title,
#                             "url": url,
#                             "content": para[:400],
#                             "source_type": "paragraph",
#                         }
#                     )

#         claim_chunks[claim] = chunks

#     return claim_chunks


# def batch_score_chunks(
#     claims: List[str], claim_chunks: Dict[str, List[Dict[str, Any]]]
# ) -> Dict[str, List[Dict[str, Any]]]:
#     """
#     Score chunks by semantic similarity to claims.
#     Batches all embeddings together for efficiency.
#     Deduplicates by title so verify node sees unique sources.
#     """
#     all_texts = []
#     text_mapping = []

#     for claim in claims:
#         all_texts.append(claim)
#         text_mapping.append((claim, "claim", None))

#         for chunk in claim_chunks.get(claim, []):
#             all_texts.append(chunk["content"][:300])
#             text_mapping.append((claim, "chunk", chunk))

#     if not all_texts:
#         return {}

#     try:
#         all_embeddings = embeddings.embed_documents(all_texts)
#     except Exception as e:
#         print(f"  Batch embedding failed: {e}")
#         return _individual_score_chunks(claims, claim_chunks)

#     claim_embeddings = {}
#     chunk_embeddings = {}

#     for i, (claim, text_type, chunk) in enumerate(text_mapping):
#         emb = np.array(all_embeddings[i])

#         if text_type == "claim":
#             claim_embeddings[claim] = emb
#         else:
#             if claim not in chunk_embeddings:
#                 chunk_embeddings[claim] = []
#             chunk_embeddings[claim].append((chunk, emb))

#     evidence = {}

#     for claim in claims:
#         claim_emb = claim_embeddings.get(claim)
#         chunks = chunk_embeddings.get(claim, [])

#         if claim_emb is None or not chunks:
#             evidence[claim] = [
#                 {
#                     "title": "No evidence found",
#                     "content": f"No Wikipedia evidence for: {claim[:100]}",
#                     "url": "",
#                     "relevance": 0.0,
#                     "source_type": "fallback",
#                 }
#             ]
#             continue

#         scored = []
#         for chunk, chunk_emb in chunks:
#             similarity = np.dot(claim_emb, chunk_emb) / (
#                 np.linalg.norm(claim_emb) * np.linalg.norm(chunk_emb) + 1e-8
#             )

#             scored.append(
#                 {
#                     "title": chunk["title"],
#                     "url": chunk.get("url", ""),
#                     "content": chunk["content"],
#                     "relevance": round(float(similarity), 4),
#                     "source_type": chunk.get("source_type", ""),
#                 }
#             )

#         scored.sort(key=lambda x: x["relevance"], reverse=True)

#         # Deduplicate by title
#         seen_titles = set()
#         deduped = []
#         for item in scored:
#             if item["title"] not in seen_titles:
#                 seen_titles.add(item["title"])
#                 deduped.append(item)

#         evidence[claim] = deduped[:2]

#     return evidence


# def _individual_score_chunks(
#     claims: List[str], claim_chunks: Dict[str, List[Dict[str, Any]]]
# ) -> Dict[str, List[Dict[str, Any]]]:
#     """
#     Fallback: Score chunks individually if batch embedding fails.
#     Also deduplicates by title.
#     """
#     evidence = {}

#     for claim in claims:
#         try:
#             claim_emb = np.array(embeddings.embed_query(claim))
#             chunks = claim_chunks.get(claim, [])

#             if not chunks:
#                 evidence[claim] = [
#                     {
#                         "title": "No evidence found",
#                         "content": "No evidence found.",
#                         "url": "",
#                         "relevance": 0.0,
#                         "source_type": "fallback",
#                     }
#                 ]
#                 continue

#             scored = []
#             for chunk in chunks:
#                 chunk_emb = np.array(embeddings.embed_query(chunk["content"][:300]))
#                 similarity = np.dot(claim_emb, chunk_emb) / (
#                     np.linalg.norm(claim_emb) * np.linalg.norm(chunk_emb) + 1e-8
#                 )
#                 scored.append({**chunk, "relevance": round(float(similarity), 4)})

#             scored.sort(key=lambda x: x["relevance"], reverse=True)

#             seen_titles = set()
#             deduped = []
#             for item in scored:
#                 if item["title"] not in seen_titles:
#                     seen_titles.add(item["title"])
#                     deduped.append(item)

#             evidence[claim] = deduped[:2]

#         except Exception as e:
#             print(f"  Scoring failed for claim: {e}")
#             evidence[claim] = [
#                 {
#                     "title": "Error",
#                     "content": "Scoring failed.",
#                     "url": "",
#                     "relevance": 0.0,
#                     "source_type": "error",
#                 }
#             ]

#     return evidence


# if __name__ == "__main__":
#     print("=" * 70)
#     print("TESTING RETRIEVAL NODE")
#     print("=" * 70)

#     test_state = {
#         "claims": [
#             "The Eiffel Tower was completed in 1889",
#             "Python was created by Guido van Rossum in 1991",
#         ]
#     }

#     start = time.time()
#     result = retrieval_node(test_state)
#     elapsed = time.time() - start

#     print(f"\nTotal time: {elapsed:.2f}s")

#     for claim, evidence_list in result["evidence"].items():
#         print(f"\nClaim: {claim}")
#         if not evidence_list:
#             print("  No evidence found")
#         for i, ev in enumerate(evidence_list, 1):
#             relevance = ev.get("relevance", 0)
#             print(f"  {i}. [{relevance:.1%}] {ev['title']}")
#             print(f"     {ev['content'][:120]}...")

"""
retrieval.py - Evidence Retrieval Node

Optimized for GTX 1650 hardware:
- Single LLM call for all query generation
- Batch Wikipedia API fetching (up to 50 pages per request)
- Persistent disk cache (never re-fetches same article)
- Batched embedding calls
- Deduplicated evidence per claim
- Defensive error handling throughout
"""

import json
import os
import re
import time
from typing import Any, Dict, List

import numpy as np
import requests
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Wikipedia API session with proper headers
_session = requests.Session()
_session.headers.update(
    {
        "User-Agent": "HallucinationRadar/1.0 (research project; direct API)",
        "Accept": "application/json",
    }
)

WIKI_API = "https://en.wikipedia.org/w/api.php"
CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "wiki_cache.json")

# Initialize models once at module level
llm = ChatOllama(model="qwen2.5:7b", temperature=0, num_ctx=2048)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# In-memory cache (loaded from disk at startup)
_article_cache: Dict[str, Any] = {}


def load_disk_cache() -> None:
    """Load persistent cache from disk into memory."""
    global _article_cache
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                _article_cache = json.load(f)
            print(f"  Loaded {len(_article_cache)} cached articles from disk")
    except Exception as e:
        print(f"  Cache load failed: {e}, starting fresh")
        _article_cache = {}


def save_disk_cache() -> None:
    """Save in-memory cache to disk."""
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(_article_cache, f, indent=2)
    except Exception as e:
        print(f"  Cache save failed: {e}")


def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve Wikipedia evidence for all claims.

    Input:  state["claims"] - list of factual claims
    Output: state["evidence"] - dict mapping claim -> list of evidence dicts
    """
    claims = state.get("claims", [])

    if not claims:
        return {"evidence": {}}

    # Load disk cache at start
    load_disk_cache()

    start_time = time.time()

    print(f"\nGenerating search queries for {len(claims)} claims...")
    all_queries = generate_all_queries(claims)

    print("Fetching Wikipedia articles...")
    claim_articles = collect_all_articles(all_queries)

    print("Extracting text chunks...")
    claim_chunks = extract_all_chunks(claims, claim_articles)

    print("Computing relevance scores...")
    evidence = batch_score_chunks(claims, claim_chunks)

    # Save updated cache to disk
    save_disk_cache()

    elapsed = time.time() - start_time
    print(f"Retrieval complete in {elapsed:.1f}s")

    return {"evidence": evidence}


def generate_all_queries(claims: List[str]) -> Dict[str, List[str]]:
    """
    Generate search queries for ALL claims in ONE LLM call.
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
- Search for the main named entity only (e.g. "GPT-3" not "GPT-3 parameters training")
- Use exact names Wikipedia would use
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
    """Extract key terms from claim for Wikipedia search."""
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


def wiki_search(query: str, num_results: int = 2) -> List[str]:
    """
    Search Wikipedia for page titles matching a query.
    Returns list of page titles.
    """
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
            r = _session.get(WIKI_API, params=params, timeout=10)

            if r.status_code == 429:
                wait = (attempt + 1) * 5
                print(
                    f"  429 on search '{query}', waiting {wait}s (attempt {attempt + 1}/4)..."
                )
                time.sleep(wait)
                continue

            r.raise_for_status()
            data = r.json()
            results = data.get("query", {}).get("search", [])
            return [item["title"] for item in results]

        except Exception as e:
            if "429" in str(e):
                wait = (attempt + 1) * 5
                print(f"  429 on search '{query}', waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Search failed '{query}': {type(e).__name__}: {str(e)[:60]}")
                return []

    print(f"  Search gave up: '{query}'")
    return []


def wiki_batch_fetch(titles: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch multiple Wikipedia pages in ONE API request.
    This is the key fix — up to 50 pages per request instead of 1.
    Returns dict mapping title -> article data.
    """
    if not titles:
        return {}

    # Wikipedia allows up to 50 titles per request pipe-separated
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
            r = _session.get(WIKI_API, params=params, timeout=30)

            if r.status_code == 429:
                wait = (attempt + 1) * 5
                print(
                    f"  429 on batch fetch, waiting {wait}s (attempt {attempt + 1}/4)..."
                )
                time.sleep(wait)
                continue

            r.raise_for_status()
            data = r.json()

            pages = data.get("query", {}).get("pages", {})
            result = {}

            for page in pages.values():
                # Skip missing pages
                if page.get("pageid", -1) == -1:
                    continue

                content = page.get("extract", "")
                if not content:
                    continue

                title = page.get("title", "Unknown")
                result[title] = {
                    "title": title,
                    "url": page.get(
                        "fullurl",
                        f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    ),
                    "summary": content[:500],
                    "paragraphs": _get_first_paragraphs(content, num_paras=2),
                    "page_id": page.get("pageid", hash(title)),
                }

            return result

        except Exception as e:
            if "429" in str(e):
                wait = (attempt + 1) * 5
                print(f"  429 on batch fetch, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Batch fetch failed: {type(e).__name__}: {str(e)[:60]}")
                return {}

    print("  Batch fetch gave up after 4 attempts")
    return {}


def collect_all_articles(
    all_queries: Dict[str, List[str]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch all unique Wikipedia articles needed.
    Uses disk cache + batch fetching to minimize API calls.
    """
    # Step 1: collect all unique queries
    unique_queries = set()
    for queries in all_queries.values():
        unique_queries.update(queries)

    # Step 2: search for titles (check cache first)
    query_to_titles: Dict[str, List[str]] = {}
    search_cache_key = lambda q: f"search:{q.lower().strip()}"

    uncached_queries = []
    for query in unique_queries:
        key = search_cache_key(query)
        if key in _article_cache:
            query_to_titles[query] = _article_cache[key]
        else:
            uncached_queries.append(query)

    # Search for uncached queries (still sequential but cached after)
    for query in uncached_queries:
        titles = wiki_search(query, num_results=2)
        query_to_titles[query] = titles
        _article_cache[search_cache_key(query)] = titles

    # Step 3: collect all unique page titles needed
    all_titles_needed = set()
    for titles in query_to_titles.values():
        all_titles_needed.update(titles)

    # Step 4: figure out which titles aren't cached yet
    uncached_titles = [t for t in all_titles_needed if t not in _article_cache]

    # Step 5: BATCH FETCH all uncached titles in as few requests as possible
    if uncached_titles:
        print(
            f"  Batch fetching {len(uncached_titles)} articles in "
            f"{(len(uncached_titles) + 49) // 50} request(s)..."
        )

        # Fetch in batches of 50
        for i in range(0, len(uncached_titles), 50):
            batch = uncached_titles[i : i + 50]
            fetched = wiki_batch_fetch(batch)

            # Store fetched articles in cache
            for title, article in fetched.items():
                _article_cache[title] = article

            # Mark failed fetches as None so we don't retry
            for title in batch:
                if title not in _article_cache:
                    _article_cache[title] = None
    else:
        print("  All articles loaded from cache!")

    # Step 6: map claims to their articles
    claim_articles = {}
    for claim, queries in all_queries.items():
        articles_for_claim = []
        seen_ids = set()

        for query in queries:
            for title in query_to_titles.get(query, []):
                article = _article_cache.get(title)
                if not article:
                    continue
                page_id = article.get("page_id")
                if page_id and page_id not in seen_ids:
                    seen_ids.add(page_id)
                    articles_for_claim.append(article)

        claim_articles[claim] = articles_for_claim[:3]

    return claim_articles


def _get_first_paragraphs(content: str, num_paras: int = 2) -> List[str]:
    """Extract first N substantive paragraphs from Wikipedia content."""
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


def extract_all_chunks(
    claims: List[str], claim_articles: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract text chunks from articles for each claim."""
    claim_chunks = {}

    for claim in claims:
        chunks = []
        articles = claim_articles.get(claim, [])

        for article in articles:
            if not article:
                continue

            title = article.get("title", "Unknown")
            url = article.get("url", "")

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
    Deduplicates by title.
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

        scored.sort(key=lambda x: x["relevance"], reverse=True)

        seen_titles = set()
        deduped = []
        for item in scored:
            if item["title"] not in seen_titles:
                seen_titles.add(item["title"])
                deduped.append(item)

        evidence[claim] = deduped[:2]

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


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING RETRIEVAL NODE")
    print("=" * 70)

    test_state = {
        "claims": [
            "The Eiffel Tower was completed in 1889",
            "Python was created by Guido van Rossum in 1991",
            "GPT-3 was developed by OpenAI",
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
            print(f"  {i}. [{relevance:.1%}] {ev['title']}")
            print(f"     {ev['content'][:120]}...")

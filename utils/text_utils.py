"""Utility functions for text processing and manipulation."""

import re
import string
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    return text


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    # Simple sentence splitter based on common punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def extract_noun_phrases(text: str) -> List[str]:
    """
    Extract simple noun phrases from text using basic pattern matching.
    
    Args:
        text: Text to extract from
        
    Returns:
        List of noun phrases
    """
    # Simple pattern for noun phrases (can be enhanced with spacy)
    # Pattern: (Adjective)* Noun+
    pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    phrases = re.findall(pattern, text)
    return phrases


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of tokens
    """
    # Remove punctuation and split
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower().split()


def calculate_overlap(text1: str, text2: str) -> float:
    """
    Calculate word overlap between two texts (Jaccard similarity).
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Overlap score between 0 and 1
    """
    tokens1 = set(tokenize(text1))
    tokens2 = set(tokenize(text2))
    
    if len(tokens1) == 0 or len(tokens2) == 0:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return intersection / union if union > 0 else 0.0


def highlight_text(text: str, phrases: List[Dict[str, any]]) -> str:
    """
    Add HTML highlighting to specific phrases in text.
    
    Args:
        text: Original text
        phrases: List of dicts with 'phrase' and 'color' keys
        
    Returns:
        HTML-formatted text with highlighting
    """
    highlighted = text
    
    for item in phrases:
        phrase = item.get('phrase', '')
        color = item.get('color', 'yellow')
        
        # Case-insensitive replacement
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        highlighted = pattern.sub(f'<mark style="background-color:{color}">{phrase}</mark>', highlighted)
    
    return highlighted


def truncate_text(text: str, max_length: int = 512) -> str:
    """
    Truncate text to maximum length while preserving complete sentences.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    sentences = split_into_sentences(text)
    result = ""
    
    for sentence in sentences:
        if len(result) + len(sentence) + 1 <= max_length:
            result += sentence + " "
        else:
            break
    
    return result.strip()


def remove_duplicates(items: List[str]) -> List[str]:
    """
    Remove duplicate items while preserving order.
    
    Args:
        items: List of items
        
    Returns:
        List with duplicates removed
    """
    seen = set()
    result = []
    
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result


def format_citation(source: str, page: int = None) -> str:
    """
    Format a citation string.
    
    Args:
        source: Source document name
        page: Optional page number
        
    Returns:
        Formatted citation
    """
    if page:
        return f"{source} (p. {page})"
    return f"{source}"


def calculate_confidence(scores: List[float]) -> float:
    """
    Calculate average confidence from a list of scores.
    
    Args:
        scores: List of confidence scores (0-1)
        
    Returns:
        Average confidence score
    """
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalize a score to a range.
    
    Args:
        score: Score to normalize
        min_val: Minimum value of target range
        max_val: Maximum value of target range
        
    Returns:
        Normalized score
    """
    return max(min_val, min(max_val, score))

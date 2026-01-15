"""Module for extracting factual claims from text."""

import re
import logging
from typing import List, Dict, Tuple
from utils.text_utils import split_into_sentences, clean_text, extract_noun_phrases
import spacy

logger = logging.getLogger(__name__)


class ClaimExtractor:
    """Extract factual claims from generated text."""
    
    def __init__(self, min_claim_length: int = 10, max_claims: int = 20):
        """
        Initialize the claim extractor.
        
        Args:
            min_claim_length: Minimum length of a claim in characters
            max_claims: Maximum number of claims to extract
        """
        self.min_claim_length = min_claim_length
        self.max_claims = max_claims
        
        # Load spaCy NLP model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Spacy model not found. Download with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract_claims(self, text: str) -> List[Dict[str, any]]:
        """
        Extract factual claims from text.
        
        Args:
            text: Text to extract claims from
            
        Returns:
            List of dicts with 'claim', 'sentence', and 'confidence' keys
        """
        claims = []
        
        # Clean text
        text = clean_text(text)
        
        # Split into sentences
        sentences = split_into_sentences(text)
        
        logger.info(f"Extracting claims from {len(sentences)} sentences")
        
        for sentence in sentences:
            if len(sentence) < self.min_claim_length:
                continue
            
            # Extract entities and relationships from sentence
            extracted = self._extract_from_sentence(sentence)
            
            if extracted:
                for claim, confidence in extracted:
                    if len(claim) >= self.min_claim_length:
                        claims.append({
                            'claim': claim,
                            'sentence': sentence,
                            'confidence': confidence,
                            'type': self._classify_claim_type(claim)
                        })
                        
                        if len(claims) >= self.max_claims:
                            logger.info(f"Reached maximum number of claims ({self.max_claims})")
                            return claims
        
        logger.info(f"Extracted {len(claims)} claims")
        return claims
    
    def _extract_from_sentence(self, sentence: str) -> List[Tuple[str, float]]:
        """
        Extract individual claims from a sentence.
        
        Args:
            sentence: Input sentence
            
        Returns:
            List of (claim, confidence) tuples
        """
        claims = []
        
        if self.nlp is None:
            # Fallback: treat entire sentence as a claim
            return [(sentence, 0.8)]
        
        # Process with spaCy
        doc = self.nlp(sentence)
        
        # Extract named entities with statements
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Extract main claims with verbs
        verbs = [token for token in doc if token.pos_ == 'VERB']
        
        # Simple claim: Subject + Verb + Object/Attribute
        if len(verbs) > 0:
            verb = verbs[0]
            
            # Find subject (nominal subject)
            subject = None
            for token in verb.lefts:
                if token.dep_ in ['nsubj', 'nsubjpass']:
                    subject = token.text
                    break
            
            # Find object or complement
            obj = None
            for token in verb.rights:
                if token.dep_ in ['dobj', 'attr', 'acomp']:
                    obj = token.text
                    break
            
            # Construct claim
            if subject and verb:
                claim = f"{subject} {verb.text}"
                if obj:
                    claim += f" {obj}"
                
                claims.append((claim, 0.8))
        
        # Add full sentence as fallback
        if not claims:
            claims.append((sentence, 0.7))
        
        return claims
    
    def _classify_claim_type(self, claim: str) -> str:
        """
        Classify the type of claim (factual, numerical, temporal, etc.).
        
        Args:
            claim: The claim text
            
        Returns:
            Type classification
        """
        claim_lower = claim.lower()
        
        # Numerical claims
        if re.search(r'\d+', claim):
            return 'numerical'
        
        # Temporal claims
        if any(word in claim_lower for word in ['was', 'is', 'will be', 'has been']):
            return 'temporal'
        
        # Comparative claims
        if any(word in claim_lower for word in ['more than', 'less than', 'greater', 'smaller']):
            return 'comparative'
        
        # Default factual
        return 'factual'
    
    def filter_claims(self, claims: List[Dict], min_confidence: float = 0.5) -> List[Dict]:
        """
        Filter claims by confidence score.
        
        Args:
            claims: List of extracted claims
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered list of claims
        """
        filtered = [c for c in claims if c.get('confidence', 0) >= min_confidence]
        logger.info(f"Filtered to {len(filtered)} claims (min confidence: {min_confidence})")
        return filtered
    
    def deduplicate_claims(self, claims: List[Dict]) -> List[Dict]:
        """
        Remove duplicate or near-duplicate claims.
        
        Args:
            claims: List of claims
            
        Returns:
            Deduplicated list
        """
        unique_claims = []
        seen_texts = set()
        
        for claim in claims:
            claim_text = claim['claim'].lower()
            
            # Check for exact duplicates
            if claim_text not in seen_texts:
                seen_texts.add(claim_text)
                unique_claims.append(claim)
        
        logger.info(f"Deduplicated to {len(unique_claims)} claims")
        return unique_claims


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    extractor = ClaimExtractor()
    
    sample_text = """
    The Earth is the third planet from the Sun. It has one natural satellite called the Moon.
    The atmosphere of Earth contains approximately 21% oxygen and 78% nitrogen.
    """
    
    claims = extractor.extract_claims(sample_text)
    
    for claim in claims:
        print(f"Claim: {claim['claim']}")
        print(f"Type: {claim['type']}")
        print(f"Confidence: {claim['confidence']}\n")

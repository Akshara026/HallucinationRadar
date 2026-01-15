"""Module for verifying claims against evidence."""

import logging
from typing import List, Dict, Tuple
from utils.text_utils import calculate_overlap, clean_text
from retireval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ClaimVerifier:
    """Verify factual claims against retrieved evidence."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        support_threshold: float = 0.7,
        conflict_threshold: float = 0.5,
        uncertainty_threshold: float = 0.4
    ):
        """
        Initialize the claim verifier.
        
        Args:
            vector_store: Vector store for document retrieval
            support_threshold: Threshold for "supported" verification
            conflict_threshold: Threshold for "conflicting" evidence
            uncertainty_threshold: Threshold for "unsupported" claims
        """
        self.vector_store = vector_store
        self.support_threshold = support_threshold
        self.conflict_threshold = conflict_threshold
        self.uncertainty_threshold = uncertainty_threshold
    
    def verify_claim(self, claim: str, top_k: int = 5) -> Dict:
        """
        Verify a single claim.
        
        Args:
            claim: Claim to verify
            top_k: Number of documents to retrieve
            
        Returns:
            Verification result dict with keys:
                - claim: The claim text
                - status: 'supported', 'partially_supported', 'unsupported', 'conflicting'
                - confidence: Confidence score (0-1)
                - evidence: List of supporting/contradicting documents
                - reasoning: Explanation of verification
        """
        logger.info(f"Verifying claim: {claim}")
        
        # Search for relevant documents
        documents = self.vector_store.search(claim, top_k=top_k, threshold=0.0)
        
        if not documents:
            logger.info("No relevant documents found")
            return {
                'claim': claim,
                'status': 'unsupported',
                'confidence': 0.0,
                'evidence': [],
                'reasoning': 'No supporting documents found in knowledge base.'
            }
        
        # Analyze evidence
        support_scores = []
        evidence = []
        
        for doc in documents:
            doc_content = clean_text(doc.get('content', ''))
            
            # Calculate different types of similarity
            semantic_similarity = doc['similarity']
            textual_overlap = calculate_overlap(claim, doc_content)
            
            # Combined score
            combined_score = 0.6 * semantic_similarity + 0.4 * textual_overlap
            
            support_scores.append(combined_score)
            
            evidence.append({
                'document_id': doc['id'],
                'title': doc.get('title', 'Unknown'),
                'source': doc.get('source', 'Unknown'),
                'similarity': float(semantic_similarity),
                'overlap': float(textual_overlap),
                'combined_score': float(combined_score)
            })
        
        # Determine verification status
        avg_score = sum(support_scores) / len(support_scores) if support_scores else 0.0
        max_score = max(support_scores) if support_scores else 0.0
        
        status, reasoning = self._determine_status(avg_score, max_score, evidence)
        
        result = {
            'claim': claim,
            'status': status,
            'confidence': max(0.0, min(1.0, avg_score)),
            'evidence': evidence,
            'reasoning': reasoning
        }
        
        logger.info(f"Verification result: {status} (confidence: {result['confidence']:.2f})")
        
        return result
    
    def verify_claims_batch(self, claims: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Verify multiple claims.
        
        Args:
            claims: List of claim dicts with 'claim' key
            top_k: Number of documents to retrieve per claim
            
        Returns:
            List of verification results
        """
        results = []
        
        for i, claim_dict in enumerate(claims, 1):
            claim_text = claim_dict.get('claim', claim_dict) if isinstance(claim_dict, dict) else claim_dict
            
            logger.info(f"Verifying claim {i}/{len(claims)}")
            
            result = self.verify_claim(claim_text, top_k)
            result['original_data'] = claim_dict
            
            results.append(result)
        
        logger.info(f"Completed verification of {len(results)} claims")
        
        return results
    
    def _determine_status(
        self,
        avg_score: float,
        max_score: float,
        evidence: List[Dict]
    ) -> Tuple[str, str]:
        """
        Determine verification status based on scores.
        
        Args:
            avg_score: Average evidence score
            max_score: Maximum evidence score
            evidence: List of evidence documents
            
        Returns:
            Tuple of (status, reasoning)
        """
        # Check for conflicting evidence
        has_conflicting = any(
            self._is_conflicting(e.get('combined_score', 0))
            for e in evidence
        )
        
        # Determine status
        if max_score >= self.support_threshold:
            if has_conflicting and avg_score < self.conflict_threshold:
                return 'conflicting', 'Found both supporting and conflicting evidence.'
            return 'supported', 'Sufficient supporting evidence found.'
        
        elif max_score >= self.uncertainty_threshold:
            return 'partially_supported', 'Found some supporting evidence but not conclusive.'
        
        else:
            return 'unsupported', 'Insufficient or no supporting evidence found.'
    
    def _is_conflicting(self, score: float) -> bool:
        """Check if evidence contradicts the claim."""
        # Simple heuristic: very low scores might indicate contradictions
        return score < self.uncertainty_threshold
    
    def get_verification_summary(self, verification_results: List[Dict]) -> Dict:
        """
        Get summary statistics from verification results.
        
        Args:
            verification_results: List of verification results
            
        Returns:
            Summary dictionary with statistics
        """
        if not verification_results:
            return {}
        
        status_counts = {}
        confidence_scores = []
        
        for result in verification_results:
            status = result.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
            confidence_scores.append(result.get('confidence', 0))
        
        total = len(verification_results)
        
        summary = {
            'total_claims': total,
            'status_distribution': {
                k: f"{v/total*100:.1f}%" for k, v in status_counts.items()
            },
            'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            'supported_count': status_counts.get('supported', 0),
            'partially_supported_count': status_counts.get('partially_supported', 0),
            'unsupported_count': status_counts.get('unsupported', 0),
            'conflicting_count': status_counts.get('conflicting', 0),
        }
        
        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage (requires initialized vector store)
    from retireval.embedder import Embedder
    
    embedder = Embedder()
    vector_store = VectorStore(embedder)
    
    # Add sample documents
    documents = [
        {
            'id': '1',
            'title': 'Earth',
            'content': 'The Earth is the third planet from the Sun.'
        }
    ]
    vector_store.add_documents(documents)
    
    verifier = ClaimVerifier(vector_store)
    
    result = verifier.verify_claim("Earth is the third planet from the Sun")
    print(f"Claim: {result['claim']}")
    print(f"Status: {result['status']}")
    print(f"Confidence: {result['confidence']:.2f}")

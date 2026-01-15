"""Module for calculating truthfulness scores."""

import logging
from typing import List, Dict, Tuple
import yaml

logger = logging.getLogger(__name__)


class TruthfulnessScorer:
    """Calculate truthfulness scores for answers based on verification results."""
    
    def __init__(self, config_path: str = "./config/settings.yaml"):
        """
        Initialize the truthfulness scorer.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        scoring_config = config.get('scoring', {})
        
        self.supported_weight = scoring_config.get('supported_weight', 1.0)
        self.partially_supported_weight = scoring_config.get('partially_supported_weight', 0.5)
        self.unsupported_weight = scoring_config.get('unsupported_weight', 0.0)
        self.hallucination_penalty = scoring_config.get('hallucination_penalty', -0.5)
    
    def calculate_score(
        self,
        verification_results: List[Dict],
        weighted: bool = True
    ) -> float:
        """
        Calculate overall truthfulness score.
        
        Args:
            verification_results: List of verification results from ClaimVerifier
            weighted: If True, weight by claim confidence; if False, treat equally
            
        Returns:
            Truthfulness score (0-100)
        """
        if not verification_results:
            return 50.0  # Neutral score if no claims
        
        total_score = 0.0
        total_weight = 0.0
        
        for result in verification_results:
            status = result.get('status', 'unsupported')
            confidence = result.get('confidence', 0.5)
            
            # Get weight for this status
            if status == 'supported':
                weight = self.supported_weight
            elif status == 'partially_supported':
                weight = self.partially_supported_weight
            elif status == 'conflicting':
                weight = self.hallucination_penalty
            else:  # unsupported
                weight = self.unsupported_weight
            
            # Apply weighting
            if weighted:
                claim_score = weight * confidence
                total_score += claim_score
                total_weight += confidence
            else:
                total_score += weight
                total_weight += 1.0
        
        # Normalize to 0-100
        if total_weight > 0:
            final_score = (total_score / total_weight) * 100
        else:
            final_score = 50.0
        
        # Clamp to 0-100
        final_score = max(0.0, min(100.0, final_score))
        
        logger.info(f"Calculated truthfulness score: {final_score:.1f}")
        
        return final_score
    
    def get_score_category(self, score: float) -> str:
        """
        Categorize a truthfulness score.
        
        Args:
            score: Truthfulness score (0-100)
            
        Returns:
            Category: 'highly_reliable', 'reliable', 'uncertain', 'unreliable', 'highly_unreliable'
        """
        if score >= 80:
            return 'highly_reliable'
        elif score >= 60:
            return 'reliable'
        elif score >= 40:
            return 'uncertain'
        elif score >= 20:
            return 'unreliable'
        else:
            return 'highly_unreliable'
    
    def get_score_description(self, score: float) -> str:
        """Get human-readable description for a score."""
        category = self.get_score_category(score)
        
        descriptions = {
            'highly_reliable': 'This answer appears to be highly reliable based on available evidence.',
            'reliable': 'This answer appears to be reliable, though some claims may need verification.',
            'uncertain': 'This answer contains claims with mixed evidence - proceed with caution.',
            'unreliable': 'This answer contains several unverified or contradicted claims.',
            'highly_unreliable': 'This answer is highly unreliable - most claims lack supporting evidence.'
        }
        
        return descriptions.get(category, 'Score could not be assessed.')
    
    def generate_report(
        self,
        answer: str,
        verification_results: List[Dict],
        score: float
    ) -> Dict:
        """
        Generate a comprehensive truthfulness report.
        
        Args:
            answer: The original answer
            verification_results: Verification results
            score: Truthfulness score
            
        Returns:
            Comprehensive report dictionary
        """
        # Categorize results
        supported = [r for r in verification_results if r.get('status') == 'supported']
        partially_supported = [r for r in verification_results if r.get('status') == 'partially_supported']
        unsupported = [r for r in verification_results if r.get('status') == 'unsupported']
        conflicting = [r for r in verification_results if r.get('status') == 'conflicting']
        
        total_claims = len(verification_results)
        
        report = {
            'answer': answer,
            'truthfulness_score': round(score, 1),
            'score_category': self.get_score_category(score),
            'description': self.get_score_description(score),
            'claim_summary': {
                'total_claims': total_claims,
                'supported': len(supported),
                'partially_supported': len(partially_supported),
                'unsupported': len(unsupported),
                'conflicting': len(conflicting),
            },
            'claim_breakdown': {
                'supported_claims': [r['claim'] for r in supported][:5],
                'partially_supported_claims': [r['claim'] for r in partially_supported][:5],
                'unsupported_claims': [r['claim'] for r in unsupported][:5],
                'conflicting_claims': [r['claim'] for r in conflicting][:5],
            },
            'risk_level': self._calculate_risk_level(score, unsupported, conflicting),
            'recommendations': self._get_recommendations(score, unsupported, conflicting)
        }
        
        return report
    
    def _calculate_risk_level(self, score: float, unsupported: List, conflicting: List) -> str:
        """Calculate overall risk level."""
        if len(conflicting) > 0 or score < 20:
            return 'high'
        elif len(unsupported) > 2 or score < 40:
            return 'medium'
        else:
            return 'low'
    
    def _get_recommendations(self, score: float, unsupported: List, conflicting: List) -> List[str]:
        """Get recommendations based on results."""
        recommendations = []
        
        if len(conflicting) > 0:
            recommendations.append('Review conflicting claims against authoritative sources.')
        
        if len(unsupported) > 3:
            recommendations.append('Many claims lack supporting evidence - verify independently.')
        
        if score < 40:
            recommendations.append('Consider consulting primary sources before relying on this answer.')
        
        if score >= 80:
            recommendations.append('This answer appears reliable based on available evidence.')
        
        if not recommendations:
            recommendations.append('Further verification recommended for critical applications.')
        
        return recommendations


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    scorer = TruthfulnessScorer()
    
    # Sample verification results
    sample_results = [
        {'claim': 'Earth is the third planet', 'status': 'supported', 'confidence': 0.95},
        {'claim': 'Moon orbits Earth', 'status': 'supported', 'confidence': 0.90},
        {'claim': 'Earth has one natural satellite', 'status': 'partially_supported', 'confidence': 0.70},
    ]
    
    score = scorer.calculate_score(sample_results)
    category = scorer.get_score_category(score)
    
    print(f"Score: {score:.1f}")
    print(f"Category: {category}")
    print(f"Description: {scorer.get_score_description(score)}")

"""Module for highlighting risky claims in text."""

import logging
import re
from typing import List, Dict, Tuple
import yaml

logger = logging.getLogger(__name__)


class Highlighter:
    """Highlight risky and uncertain sentences in text."""
    
    def __init__(self, config_path: str = "./config/settings.yaml"):
        """
        Initialize the highlighter.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        highlighting_config = config.get('highlighting', {})
        
        self.high_risk_threshold = highlighting_config.get('high_risk_threshold', 0.3)
        self.medium_risk_threshold = highlighting_config.get('medium_risk_threshold', 0.6)
        self.colors = highlighting_config.get('color_scheme', {
            'high_risk': 'red',
            'medium_risk': 'orange',
            'low_risk': 'green'
        })
    
    def highlight_answer(
        self,
        answer: str,
        verification_results: List[Dict]
    ) -> Dict:
        """
        Create highlighted version of answer with risk markers.
        
        Args:
            answer: Original answer text
            verification_results: Verification results from ClaimVerifier
            
        Returns:
            Dict with:
                - original: Original text
                - highlighted_html: HTML with highlighting
                - risk_map: Mapping of sentences to risk levels
                - summary: Summary of risky areas
        """
        # Split into sentences
        sentences = self._split_sentences(answer)
        
        # Map verification results to sentences
        risk_map = self._map_risks_to_sentences(sentences, verification_results)
        
        # Create HTML highlighting
        highlighted_html = self._create_highlighted_html(sentences, risk_map)
        
        # Create summary
        summary = self._create_summary(risk_map)
        
        return {
            'original': answer,
            'highlighted_html': highlighted_html,
            'risk_map': risk_map,
            'summary': summary
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _map_risks_to_sentences(
        self,
        sentences: List[str],
        verification_results: List[Dict]
    ) -> Dict[int, Dict]:
        """Map risk levels to sentences."""
        risk_map = {}
        
        for i, sentence in enumerate(sentences):
            risk_map[i] = {
                'sentence': sentence,
                'risk_level': 'low',
                'risk_score': 1.0,
                'related_claims': []
            }
        
        # Match verification results to sentences
        for result in verification_results:
            claim = result.get('claim', '')
            status = result.get('status', 'unsupported')
            confidence = result.get('confidence', 0.5)
            
            # Calculate risk score
            if status == 'supported':
                risk_score = 0.9
                risk_level = 'low'
            elif status == 'partially_supported':
                risk_score = 0.6
                risk_level = 'medium'
            elif status == 'conflicting':
                risk_score = 0.2
                risk_level = 'high'
            else:  # unsupported
                risk_score = 0.3
                risk_level = 'high'
            
            # Adjust by confidence
            risk_score *= confidence
            
            # Find matching sentences
            for i, sentence in enumerate(sentences):
                if self._claim_in_sentence(claim, sentence):
                    # Update risk level
                    if risk_score < risk_map[i]['risk_score']:
                        risk_map[i]['risk_score'] = risk_score
                        risk_map[i]['risk_level'] = risk_level
                    
                    risk_map[i]['related_claims'].append({
                        'claim': claim,
                        'status': status,
                        'confidence': confidence
                    })
        
        return risk_map
    
    def _claim_in_sentence(self, claim: str, sentence: str) -> bool:
        """Check if claim is in sentence (simple substring matching)."""
        # Convert to lowercase for comparison
        claim_lower = claim.lower()
        sentence_lower = sentence.lower()
        
        # Check for substring match
        if claim_lower in sentence_lower:
            return True
        
        # Check for word overlap
        claim_words = set(claim_lower.split())
        sentence_words = set(sentence_lower.split())
        
        overlap = len(claim_words & sentence_words)
        
        # If significant overlap, consider it a match
        return overlap >= len(claim_words) * 0.7
    
    def _create_highlighted_html(
        self,
        sentences: List[str],
        risk_map: Dict[int, Dict]
    ) -> str:
        """Create HTML with risk highlighting."""
        html_parts = []
        
        for i, sentence in enumerate(sentences):
            risk_info = risk_map.get(i, {})
            risk_level = risk_info.get('risk_level', 'low')
            
            # Get color
            color = self.colors.get(f'{risk_level}_risk', 'white')
            
            # Create highlight
            if risk_level == 'low':
                highlighted_sentence = f'<span>{sentence}</span>'
            else:
                highlighted_sentence = f'<mark style="background-color: {color}; padding: 2px; border-radius: 3px;">{sentence}</mark>'
            
            html_parts.append(highlighted_sentence)
        
        html = ' '.join(html_parts)
        
        return f'<p>{html}</p>'
    
    def _create_summary(self, risk_map: Dict[int, Dict]) -> Dict:
        """Create summary of risky areas."""
        high_risk_sentences = []
        medium_risk_sentences = []
        
        for i, risk_info in risk_map.items():
            risk_level = risk_info.get('risk_level')
            sentence = risk_info.get('sentence', '')
            
            if risk_level == 'high':
                high_risk_sentences.append(sentence)
            elif risk_level == 'medium':
                medium_risk_sentences.append(sentence)
        
        return {
            'high_risk_count': len(high_risk_sentences),
            'medium_risk_count': len(medium_risk_sentences),
            'high_risk_sentences': high_risk_sentences[:3],
            'medium_risk_sentences': medium_risk_sentences[:3]
        }
    
    def create_annotated_json(
        self,
        answer: str,
        verification_results: List[Dict]
    ) -> Dict:
        """
        Create JSON annotation of answer with risk markers.
        
        Args:
            answer: Original answer
            verification_results: Verification results
            
        Returns:
            JSON-serializable dictionary with annotations
        """
        highlighted = self.highlight_answer(answer, verification_results)
        
        sentences = highlighted['risk_map']
        
        annotations = []
        for sent_idx, risk_info in sentences.items():
            annotations.append({
                'sentence_index': sent_idx,
                'sentence': risk_info['sentence'],
                'risk_level': risk_info['risk_level'],
                'risk_score': round(risk_info['risk_score'], 2),
                'related_claims': risk_info['related_claims']
            })
        
        return {
            'answer': answer,
            'annotations': annotations,
            'summary': highlighted['summary']
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    highlighter = Highlighter()
    
    answer = "The Earth is round and has one moon. The moon is made of cheese."
    
    verification_results = [
        {
            'claim': 'Earth is round',
            'status': 'supported',
            'confidence': 0.95
        },
        {
            'claim': 'moon is made of cheese',
            'status': 'unsupported',
            'confidence': 0.5
        }
    ]
    
    highlighted = highlighter.highlight_answer(answer, verification_results)
    
    print("Original:", highlighted['original'])
    print("\nHighlighted HTML:")
    print(highlighted['highlighted_html'])
    print("\nRisk Summary:")
    print(highlighted['summary'])

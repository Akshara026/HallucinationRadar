"""Main orchestration script for the HallucinationRadar pipeline."""

import logging
import sys
import yaml
from typing import Dict, List, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
from llm.answer_generator import AnswerGenerator
from claims.claim_extractor import ClaimExtractor
from retireval.doc_loader import DocumentLoader
from retireval.embedder import Embedder
from retireval.vector_store import VectorStore
from verification.claim_verifier import ClaimVerifier
from scoring.truthfulness import TruthfulnessScorer
from highlighting.highlighter import Highlighter


class HallucinationRadar:
    """Main orchestrator for hallucination detection pipeline."""
    
    def __init__(self, config_path: str = "./config/settings.yaml"):
        """
        Initialize HallucinationRadar pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("Initializing HallucinationRadar...")
        
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        logger.info("Loading LLM...")
        self.answer_generator = AnswerGenerator(config_path)
        
        logger.info("Loading claim extractor...")
        claims_config = self.config.get('claims', {})
        self.claim_extractor = ClaimExtractor(
            min_claim_length=claims_config.get('min_claim_length', 10),
            max_claims=claims_config.get('max_claims_per_answer', 20)
        )
        
        logger.info("Loading document retrieval system...")
        self.doc_loader = DocumentLoader(
            docs_path=self.config.get('data', {}).get('documents_path', './data/documents')
        )
        
        logger.info("Loading embedder...")
        embedding_config = self.config.get('embedding', {})
        self.embedder = Embedder(
            model_name=embedding_config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
            device=embedding_config.get('device', 'cpu')
        )
        
        logger.info("Initializing vector store...")
        self.vector_store = VectorStore(self.embedder)
        
        logger.info("Loading verification system...")
        verification_config = self.config.get('verification', {})
        self.claim_verifier = ClaimVerifier(
            vector_store=self.vector_store,
            support_threshold=verification_config.get('support_threshold', 0.7),
            conflict_threshold=verification_config.get('conflict_threshold', 0.5),
            uncertainty_threshold=verification_config.get('uncertainty_threshold', 0.4)
        )
        
        logger.info("Loading scoring system...")
        self.truthfulness_scorer = TruthfulnessScorer(config_path)
        
        logger.info("Loading highlighter...")
        self.highlighter = Highlighter(config_path)
        
        self.documents_loaded = False
        logger.info("HallucinationRadar initialized successfully!")
    
    def load_documents(self) -> bool:
        """Load documents for evidence retrieval."""
        logger.info("Loading documents...")
        
        documents = self.doc_loader.load_documents()
        
        if documents:
            self.vector_store.add_documents(documents)
            self.documents_loaded = True
            logger.info(f"Successfully loaded {len(documents)} documents")
            return True
        else:
            logger.warning("No documents loaded. Add PDFs/TXT files to data/documents/")
            return False
    
    def verify_answer(self, question: str, answer: str) -> Dict:
        """
        Verify an answer for hallucinations.
        
        Args:
            question: User's question
            answer: Generated or provided answer to verify
            
        Returns:
            Comprehensive verification report
        """
        logger.info(f"Verifying answer for question: {question}")
        
        if not self.documents_loaded:
            logger.warning("No documents loaded. Running with limited verification.")
        
        # Step 1: Extract claims
        logger.info("Step 1: Extracting claims from answer...")
        claims = self.claim_extractor.extract_claims(answer)
        
        if not claims:
            logger.warning("No claims extracted from answer")
            return {
                'question': question,
                'answer': answer,
                'claims': [],
                'truthfulness_score': 50.0,
                'error': 'No claims could be extracted'
            }
        
        logger.info(f"Extracted {len(claims)} claims")
        
        # Step 2: Verify claims
        logger.info("Step 2: Verifying claims against documents...")
        verification_results = self.claim_verifier.verify_claims_batch(claims)
        
        # Step 3: Calculate truthfulness score
        logger.info("Step 3: Calculating truthfulness score...")
        truthfulness_score = self.truthfulness_scorer.calculate_score(verification_results)
        
        # Step 4: Generate report
        logger.info("Step 4: Generating report...")
        report = self.truthfulness_scorer.generate_report(answer, verification_results, truthfulness_score)
        
        # Step 5: Highlight risky claims
        logger.info("Step 5: Creating highlighted version...")
        highlighted = self.highlighter.highlight_answer(answer, verification_results)
        
        # Combine into final result
        result = {
            'question': question,
            'answer': answer,
            'claims': claims,
            'verification_results': verification_results,
            'truthfulness_score': truthfulness_score,
            'report': report,
            'highlighted_html': highlighted['highlighted_html'],
            'risk_summary': highlighted['summary']
        }
        
        logger.info(f"Verification complete. Truthfulness score: {truthfulness_score:.1f}")
        
        return result
    
    def generate_and_verify(self, question: str) -> Dict:
        """
        Generate an answer and verify it in one pipeline.
        
        Args:
            question: User's question
            
        Returns:
            Verification report with generated answer
        """
        logger.info(f"Generating and verifying answer for: {question}")
        
        # Generate answer
        logger.info("Generating answer...")
        answer = self.answer_generator.generate_answer(question)
        
        # Verify answer
        result = self.verify_answer(question, answer)
        
        return result
    
    def batch_verify(self, questions_and_answers: List[Dict]) -> List[Dict]:
        """
        Verify multiple questions and answers.
        
        Args:
            questions_and_answers: List of dicts with 'question' and 'answer' keys
            
        Returns:
            List of verification reports
        """
        results = []
        
        for i, item in enumerate(questions_and_answers, 1):
            logger.info(f"Processing {i}/{len(questions_and_answers)}")
            
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            result = self.verify_answer(question, answer)
            results.append(result)
        
        return results


def main():
    """Main entry point."""
    # Initialize pipeline
    radar = HallucinationRadar()
    
    # Load documents
    radar.load_documents()
    
    # Example usage
    question = "What is the capital of France?"
    
    # Option 1: Verify provided answer
    answer = "Paris is the capital of France, known as the City of Light."
    result = radar.verify_answer(question, answer)
    
    print("\n" + "="*80)
    print("VERIFICATION RESULT")
    print("="*80)
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Truthfulness Score: {result['truthfulness_score']:.1f}")
    print(f"Category: {result['report']['score_category']}")
    print(f"\nDescription: {result['report']['description']}")
    
    print(f"\nClaim Summary:")
    for claim in result['report']['claim_breakdown']['supported_claims'][:3]:
        print(f"  ✓ {claim}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

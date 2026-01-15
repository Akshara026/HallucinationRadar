"""Evaluation module for testing HallucinationRadar performance."""

import logging
import json
from typing import List, Dict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class FEVEREvaluator:
    """
    Evaluate hallucination detection against FEVER-style benchmarks.
    
    FEVER (Fact Extraction and VERification) is a benchmark dataset for 
    fact verification and hallucination detection.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        logger.info("Initializing FEVER Evaluator")
    
    def load_fever_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load FEVER dataset from file.
        
        Args:
            dataset_path: Path to FEVER dataset JSON file
            
        Returns:
            List of FEVER samples with 'claim', 'evidence', 'label' keys
        """
        logger.info(f"Loading FEVER dataset from {dataset_path}")
        
        samples = []
        
        try:
            with open(dataset_path, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    samples.append(sample)
            
            logger.info(f"Loaded {len(samples)} samples")
        
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
        
        return samples
    
    def evaluate_predictions(
        self,
        predictions: List[Dict],
        labels: List[str]
    ) -> Dict:
        """
        Evaluate model predictions against ground truth.
        
        Args:
            predictions: List of model predictions with 'status' key
            labels: Ground truth labels
            
        Returns:
            Evaluation metrics dictionary
        """
        if len(predictions) != len(labels):
            raise ValueError("Predictions and labels length mismatch")
        
        correct = 0
        total = len(predictions)
        
        for pred, label in zip(predictions, labels):
            if pred.get('status') == label:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
        
        return metrics
    
    def calculate_f1_score(
        self,
        predictions: List[Dict],
        labels: List[str],
        label_type: str
    ) -> float:
        """
        Calculate F1 score for a specific label.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            label_type: Label to calculate F1 for
            
        Returns:
            F1 score
        """
        tp = sum(1 for p, l in zip(predictions, labels) 
                 if p.get('status') == label_type and l == label_type)
        fp = sum(1 for p, l in zip(predictions, labels) 
                 if p.get('status') == label_type and l != label_type)
        fn = sum(1 for p, l in zip(predictions, labels) 
                 if p.get('status') != label_type and l == label_type)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1
    
    def generate_report(
        self,
        results: List[Dict],
        output_path: str = "./evaluation_report.json"
    ):
        """
        Generate evaluation report.
        
        Args:
            results: Evaluation results
            output_path: Path to save report
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Report saved to {output_path}")
        
        except Exception as e:
            logger.error(f"Error saving report: {e}")


class TruthfulnessEvaluator:
    """Evaluate truthfulness scores."""
    
    def __init__(self):
        """Initialize evaluator."""
        logger.info("Initializing Truthfulness Evaluator")
    
    def evaluate_score_correlation(
        self,
        predicted_scores: List[float],
        true_labels: List[int]
    ) -> Dict:
        """
        Evaluate correlation between predicted scores and true labels.
        
        Args:
            predicted_scores: Predicted truthfulness scores (0-100)
            true_labels: Ground truth labels (0=false, 1=true)
            
        Returns:
            Correlation metrics
        """
        if len(predicted_scores) != len(true_labels):
            raise ValueError("Scores and labels length mismatch")
        
        # Normalize scores to 0-1
        normalized_scores = [s / 100.0 for s in predicted_scores]
        
        # Calculate correlation
        mean_score = sum(normalized_scores) / len(normalized_scores)
        mean_label = sum(true_labels) / len(true_labels)
        
        numerator = sum(
            (s - mean_score) * (l - mean_label)
            for s, l in zip(normalized_scores, true_labels)
        )
        
        denominator_s = sum((s - mean_score) ** 2 for s in normalized_scores)
        denominator_l = sum((l - mean_label) ** 2 for l in true_labels)
        
        correlation = numerator / (denominator_s * denominator_l) ** 0.5 if denominator_s > 0 and denominator_l > 0 else 0.0
        
        return {
            'pearson_correlation': correlation,
            'mean_predicted_score': sum(predicted_scores) / len(predicted_scores),
            'mean_true_label': mean_label
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example evaluation
    evaluator = FEVEREvaluator()
    
    # Mock predictions and labels
    predictions = [
        {'status': 'supported'},
        {'status': 'unsupported'},
        {'status': 'partially_supported'}
    ]
    
    labels = ['supported', 'unsupported', 'unsupported']
    
    metrics = evaluator.evaluate_predictions(predictions, labels)
    print(f"Accuracy: {metrics['accuracy']:.2%}")

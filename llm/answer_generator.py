"""LLM module for generating answers using pre-trained language models."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import Optional
import yaml

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generate answers using a pre-trained language model."""
    
    def __init__(self, config_path: str = "./config/settings.yaml", device: Optional[str] = None):
        """
        Initialize the answer generator.
        
        Args:
            config_path: Path to configuration file
            device: Device to use ('cpu' or 'cuda'), defaults to auto-detection
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.llm_config = config.get('llm', {})
        self.model_name = self.llm_config.get('model_name', 'gpt2')
        self.max_length = self.llm_config.get('max_length', 256)
        self.temperature = self.llm_config.get('temperature', 0.7)
        self.top_p = self.llm_config.get('top_p', 0.9)
        
        logger.info(f"Loading model: {self.model_name} on device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def generate_answer(
        self,
        question: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        Generate an answer to a question.
        
        Args:
            question: The question to answer
            max_new_tokens: Maximum tokens to generate (defaults to config)
            temperature: Sampling temperature (defaults to config)
            top_p: Nucleus sampling parameter (defaults to config)
            
        Returns:
            Generated answer text
        """
        max_new_tokens = max_new_tokens or self.max_length
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        
        logger.info(f"Generating answer for: {question[:100]}...")
        
        # Prepare input
        input_ids = self.tokenizer.encode(question, return_tensors='pt').to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the full output
        full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove the input question)
        answer = full_output[len(question):].strip()
        
        logger.info(f"Generated answer: {answer[:100]}...")
        
        return answer if answer else full_output
    
    def generate_answers_batch(
        self,
        questions: list,
        max_new_tokens: Optional[int] = None
    ) -> list:
        """
        Generate answers for multiple questions.
        
        Args:
            questions: List of questions
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of generated answers
        """
        answers = []
        for question in questions:
            answer = self.generate_answer(question, max_new_tokens)
            answers.append(answer)
        
        return answers
    
    def set_device(self, device: str):
        """
        Change the device the model runs on.
        
        Args:
            device: 'cpu' or 'cuda'
        """
        if device not in ['cpu', 'cuda']:
            raise ValueError("Device must be 'cpu' or 'cuda'")
        
        self.device = device
        self.model.to(device)
        logger.info(f"Model moved to {device}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    generator = AnswerGenerator()
    
    question = "What is the capital of France?"
    answer = generator.generate_answer(question)
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")

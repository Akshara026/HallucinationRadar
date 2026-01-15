"""Module for creating text embeddings using sentence transformers."""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Union, Tuple
import pickle
import os

logger = logging.getLogger(__name__)


class Embedder:
    """Create embeddings for texts using sentence transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading embedding model: {model_name} on device: {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (1D numpy array)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Embedding matrix (2D numpy array)
        """
        logger.info(f"Embedding {len(texts)} texts with batch size {batch_size}")
        embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
        return embeddings
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between -1 and 1 (typically 0 to 1)
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def find_similar(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[int, float]]:
        """
        Find similar documents for a query embedding.
        
        Args:
            query_embedding: Query embedding
            document_embeddings: Matrix of document embeddings
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (document_index, similarity_score) tuples, sorted by similarity
        """
        # Calculate cosine similarity for all documents
        similarities = []
        
        for i, doc_embedding in enumerate(document_embeddings):
            sim = self.cosine_similarity(query_embedding, doc_embedding)
            
            if sim >= threshold:
                similarities.append((i, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return similarities[:top_k]
    
    def save_embeddings(self, embeddings: np.ndarray, path: str):
        """Save embeddings to file."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, embeddings)
            logger.info(f"Saved embeddings to {path}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
    
    def load_embeddings(self, path: str) -> np.ndarray:
        """Load embeddings from file."""
        try:
            embeddings = np.load(path)
            logger.info(f"Loaded embeddings from {path}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return None
    
    def set_device(self, device: str):
        """Change the device."""
        if device not in ['cpu', 'cuda']:
            raise ValueError("Device must be 'cpu' or 'cuda'")
        
        self.device = device
        self.model.to(device)
        logger.info(f"Embedder moved to {device}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    embedder = Embedder()
    
    texts = [
        "The Earth is the third planet from the Sun.",
        "Paris is the capital of France.",
        "Dogs are mammals."
    ]
    
    embeddings = embedder.embed_texts(texts)
    print(f"Created {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    
    # Find similar
    query = "Our planet orbits the sun"
    query_embedding = embedder.embed_text(query)
    
    similar = embedder.find_similar(query_embedding, embeddings, top_k=2)
    print(f"\nMost similar to '{query}':")
    for idx, sim in similar:
        print(f"  {texts[idx]} (similarity: {sim:.4f})")

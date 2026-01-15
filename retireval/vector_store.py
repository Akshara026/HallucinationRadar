"""Vector store for managing document embeddings and similarity search."""

import numpy as np
import logging
import json
import os
from typing import List, Dict, Tuple, Optional
from retireval.embedder import Embedder
from retireval.doc_loader import DocumentLoader

logger = logging.getLogger(__name__)


class VectorStore:
    """Manage document embeddings and retrieve similar documents."""
    
    def __init__(
        self,
        embedder: Embedder,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        """
        Initialize the vector store.
        
        Args:
            embedder: Embedder instance
            embedding_model: Name of embedding model (for reference)
            device: Device to use
        """
        self.embedder = embedder
        self.embedding_model = embedding_model
        self.device = device
        
        self.documents = []
        self.embeddings = None
        self.doc_ids = []
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Add documents to the store and create embeddings.
        
        Args:
            documents: List of document dicts with 'id', 'title', 'content' keys
        """
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        self.documents = documents
        self.doc_ids = [doc['id'] for doc in documents]
        
        # Extract content for embedding
        contents = [doc.get('content', '') for doc in documents]
        
        # Create embeddings
        logger.info("Creating embeddings for documents...")
        self.embeddings = self.embedder.embed_texts(contents, batch_size=32)
        
        logger.info(f"Vector store now contains {len(self.documents)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3
    ) -> List[Dict]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of top results
            threshold: Minimum similarity threshold
            
        Returns:
            List of relevant documents with similarity scores
        """
        if not self.documents or self.embeddings is None:
            logger.warning("Vector store is empty")
            return []
        
        logger.info(f"Searching for: {query}")
        
        # Embed query
        query_embedding = self.embedder.embed_text(query)
        
        # Find similar documents
        similar = self.embedder.find_similar(
            query_embedding,
            self.embeddings,
            top_k=top_k,
            threshold=threshold
        )
        
        # Build result
        results = []
        for doc_idx, similarity in similar:
            doc = self.documents[doc_idx].copy()
            doc['similarity'] = float(similarity)
            results.append(doc)
        
        logger.info(f"Found {len(results)} relevant documents")
        return results
    
    def search_multiple_queries(
        self,
        queries: List[str],
        top_k: int = 5,
        threshold: float = 0.3
    ) -> List[List[Dict]]:
        """
        Search for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of top results per query
            threshold: Minimum similarity threshold
            
        Returns:
            List of result lists (one per query)
        """
        results = []
        for query in queries:
            result = self.search(query, top_k, threshold)
            results.append(result)
        
        return results
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Get a specific document by ID."""
        for doc in self.documents:
            if doc['id'] == doc_id:
                return doc
        return None
    
    def save_state(self, path: str):
        """Save vector store state (documents, embeddings, IDs)."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            state = {
                'documents': self.documents,
                'doc_ids': self.doc_ids,
                'embedding_model': self.embedding_model
            }
            
            # Save metadata as JSON
            metadata_path = path.replace('.npz', '.json')
            with open(metadata_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Save embeddings as numpy
            embeddings_path = path.replace('.npz', '.npy')
            if self.embeddings is not None:
                np.save(embeddings_path, self.embeddings)
            
            logger.info(f"Saved vector store to {path}")
        
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    def load_state(self, path: str) -> bool:
        """Load vector store state."""
        try:
            # Load metadata
            metadata_path = path.replace('.npz', '.json')
            with open(metadata_path, 'r') as f:
                state = json.load(f)
            
            self.documents = state['documents']
            self.doc_ids = state['doc_ids']
            
            # Load embeddings
            embeddings_path = path.replace('.npz', '.npy')
            if os.path.exists(embeddings_path):
                self.embeddings = np.load(embeddings_path)
            
            logger.info(f"Loaded vector store from {path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def clear(self):
        """Clear all documents and embeddings."""
        self.documents = []
        self.embeddings = None
        self.doc_ids = []
        logger.info("Vector store cleared")
    
    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        stats = {
            'num_documents': len(self.documents),
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'total_embeddings': len(self.embeddings) if self.embeddings is not None else 0,
        }
        return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize
    embedder = Embedder()
    vector_store = VectorStore(embedder)
    
    # Add sample documents
    documents = [
        {
            'id': '1',
            'title': 'Earth',
            'content': 'The Earth is the third planet from the Sun in our solar system.'
        },
        {
            'id': '2',
            'title': 'Moon',
            'content': 'The Moon is Earth\'s natural satellite and has a diameter of 3,474 km.'
        },
        {
            'id': '3',
            'title': 'Mars',
            'content': 'Mars is the fourth planet from the Sun and is known as the red planet.'
        }
    ]
    
    vector_store.add_documents(documents)
    
    # Search
    results = vector_store.search("Tell me about planets", top_k=2)
    
    for result in results:
        print(f"ID: {result['id']}")
        print(f"Title: {result['title']}")
        print(f"Similarity: {result['similarity']:.4f}\n")

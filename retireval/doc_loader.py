"""Module for loading and managing documents for evidence retrieval."""

import os
import logging
from typing import List, Dict
import json
import pickle
import pdfplumber
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load documents from various sources."""
    
    def __init__(self, docs_path: str = "./data/documents"):
        """
        Initialize the document loader.
        
        Args:
            docs_path: Path to documents directory
        """
        self.docs_path = docs_path
        self.documents = []
    
    def load_documents(self) -> List[Dict[str, str]]:
        """
        Load all documents from the documents directory.
        
        Returns:
            List of documents with 'id', 'title', 'content', and 'source' keys
        """
        self.documents = []
        
        if not os.path.exists(self.docs_path):
            logger.warning(f"Documents path does not exist: {self.docs_path}")
            return []
        
        # Load PDF files
        pdf_files = list(Path(self.docs_path).glob('*.pdf'))
        for pdf_file in pdf_files:
            docs = self._load_pdf(str(pdf_file))
            self.documents.extend(docs)
        
        # Load text files
        txt_files = list(Path(self.docs_path).glob('*.txt'))
        for txt_file in txt_files:
            docs = self._load_text_file(str(txt_file))
            self.documents.extend(docs)
        
        # Load JSON files
        json_files = list(Path(self.docs_path).glob('*.json'))
        for json_file in json_files:
            docs = self._load_json_file(str(json_file))
            self.documents.extend(docs)
        
        logger.info(f"Loaded {len(self.documents)} documents")
        return self.documents
    
    def _load_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """Load content from PDF file."""
        documents = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    
                    documents.append({
                        'id': f"{os.path.basename(pdf_path)}_p{page_num}",
                        'title': f"{os.path.basename(pdf_path)} - Page {page_num}",
                        'content': text,
                        'source': pdf_path,
                        'page': page_num
                    })
            
            logger.info(f"Loaded {len(documents)} pages from {os.path.basename(pdf_path)}")
        
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
        
        return documents
    
    def _load_text_file(self, txt_path: str) -> List[Dict[str, str]]:
        """Load content from text file."""
        documents = []
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            documents.append({
                'id': os.path.basename(txt_path),
                'title': os.path.basename(txt_path),
                'content': content,
                'source': txt_path,
                'page': None
            })
            
            logger.info(f"Loaded text file {os.path.basename(txt_path)}")
        
        except Exception as e:
            logger.error(f"Error loading text file {txt_path}: {e}")
        
        return documents
    
    def _load_json_file(self, json_path: str) -> List[Dict[str, str]]:
        """Load documents from JSON file."""
        documents = []
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Expect list of document objects or single document
            if isinstance(data, list):
                for i, doc in enumerate(data):
                    doc['id'] = doc.get('id', f"{os.path.basename(json_path)}_{i}")
                    doc['source'] = json_path
                    documents.append(doc)
            else:
                data['id'] = data.get('id', os.path.basename(json_path))
                data['source'] = json_path
                documents.append(data)
            
            logger.info(f"Loaded {len(documents)} documents from {os.path.basename(json_path)}")
        
        except Exception as e:
            logger.error(f"Error loading JSON file {json_path}: {e}")
        
        return documents
    
    def get_documents(self) -> List[Dict[str, str]]:
        """Get loaded documents."""
        return self.documents
    
    def save_documents_cache(self, cache_path: str):
        """Save documents to cache."""
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.documents, f)
            logger.info(f"Documents cached to {cache_path}")
        except Exception as e:
            logger.error(f"Error saving documents cache: {e}")
    
    def load_documents_cache(self, cache_path: str) -> bool:
        """Load documents from cache."""
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"Loaded {len(self.documents)} documents from cache")
                return True
        except Exception as e:
            logger.error(f"Error loading documents cache: {e}")
        
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    loader = DocumentLoader()
    docs = loader.load_documents()
    
    for doc in docs[:3]:
        print(f"ID: {doc['id']}")
        print(f"Title: {doc['title']}")
        print(f"Content: {doc['content'][:100]}...\n")

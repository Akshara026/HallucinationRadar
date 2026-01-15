# HallucinationRadar - Complete Project Index

## 📚 Documentation Files

### Getting Started

- **README.md** - Full project documentation with features, installation, usage
- **QUICKSTART.py** - Interactive quick start examples and code samples
- **PROJECT_COMPLETION.md** - Project completion summary and architecture
- **BEST_PRACTICES.md** - Best practices, patterns, and troubleshooting

## 🏗️ Project Architecture

### Core Modules

#### 1. Configuration & Settings

- **config/settings.yaml** - Central configuration file
  - LLM settings (model, temperature, max_length)
  - Embedding configuration
  - Verification thresholds
  - Scoring weights
  - Highlighting colors

#### 2. LLM & Generation

- **llm/answer_generator.py** - Answer generation using transformer models
  - Load and manage LLM models
  - Generate answers from questions
  - Batch processing support
  - GPU/CPU device management

#### 3. Claim Extraction

- **claims/claim_extractor.py** - Extract factual claims from text
  - spaCy-based NLP processing
  - Entity and relationship extraction
  - Claim type classification
  - Deduplication and filtering

#### 4. Document Retrieval

- **retireval/doc_loader.py** - Load documents from various formats
  - PDF support (pdfplumber)
  - Text file support
  - JSON file support
  - Caching mechanism
- **retireval/embedder.py** - Create text embeddings
  - Sentence Transformers integration
  - Cosine similarity calculation
  - Batch embedding support
  - Embedding persistence
- **retireval/vector_store.py** - Semantic search and storage
  - Add documents with embeddings
  - Search functionality
  - Similarity retrieval
  - State persistence
- **retireval/web_search.py** - Web search integration (extensible)
  - Placeholder for web search APIs
  - Search result processing

#### 5. Claim Verification

- **verification/claim_verifier.py** - Verify claims against evidence
  - Document retrieval and evidence collection
  - Semantic and textual similarity analysis
  - Status determination (supported/unsupported/etc.)
  - Batch verification
  - Summary generation

#### 6. Scoring & Analysis

- **scoring/truthfulness.py** - Calculate truthfulness scores
  - Weighted score calculation
  - Score categorization
  - Comprehensive reporting
  - Risk assessment
  - Recommendations generation

#### 7. Highlighting & Visualization

- **highlighting/highlighter.py** - Mark risky claims in text
  - Sentence-level risk mapping
  - HTML visualization
  - Risk color coding
  - Risk summarization
  - JSON annotation export

#### 8. Utilities

- **utils/text_utils.py** - Text processing utilities
  - Text cleaning and normalization
  - Sentence splitting
  - Tokenization
  - Overlap calculation
  - HTML highlighting
  - Citation formatting

#### 9. Evaluation

- **evaluation/fever_eval.py** - Evaluation framework
  - FEVER dataset support
  - Accuracy metrics
  - F1 score calculation
  - Correlation analysis
  - Report generation

### Main Entry Points

#### 1. CLI/Python API

- **main.py** - Main orchestration script
  - HallucinationRadar class (main interface)
  - verify_answer() - Verify single answer
  - generate_and_verify() - Generate and verify
  - batch_verify() - Process multiple items
  - load_documents() - Initialize knowledge base

#### 2. Web Interface

- **app.py** - Gradio web interface
  - Tab 1: Verify Answer
  - Tab 2: Generate & Verify
  - Tab 3: Batch Verification
  - Tab 4: About/Help

## 🔄 Data Flow

```
User Input (Question + Answer)
          ↓
    main.py (Orchestration)
          ↓
   ┌──────┴──────┐
   ↓             ↓
Claim          Evidence
Extraction      Retrieval
   ↓             ↓
claims/      retireval/
extractor    (doc_loader,
             embedder,
             vector_store)
             ↓
          Verification
             ↓
       verification/
       claim_verifier
             ↓
          Scoring
             ↓
       scoring/
       truthfulness
             ↓
       Highlighting
             ↓
       highlighting/
       highlighter
             ↓
    Final Report Output
```

## 📊 Processing Pipeline

1. **Input**: User provides question + answer
2. **Claim Extraction**: Extract factual statements using NLP
3. **Document Loading**: Load knowledge base documents
4. **Embedding Creation**: Create embeddings for documents
5. **Evidence Retrieval**: Find relevant documents for each claim
6. **Claim Verification**: Compare claims against evidence
7. **Score Calculation**: Aggregate verification results
8. **Risk Assessment**: Determine risk levels
9. **Output Generation**: Create comprehensive report
10. **Visualization**: Highlight risky areas in HTML

## 🎯 Key Features by Module

### Answer Generation

- Multiple LLM model support
- Configurable generation parameters
- Batch processing
- Device optimization

### Claim Extraction

- NLP-based extraction
- Multiple claim types
- Confidence scoring
- Deduplication

### Evidence Retrieval

- Multi-format document support
- Semantic search
- Vector caching
- Batch retrieval

### Verification

- Semantic similarity
- Textual overlap
- Evidence weighting
- Confidence calculation

### Scoring

- Weighted aggregation
- Categorization (5 levels)
- Comprehensive reporting
- Risk assessment

### Visualization

- HTML highlighting
- Color-coded risk levels
- Interactive web interface
- JSON export

## ⚙️ Configuration Parameters

### LLM Settings

- `model_name`: HuggingFace model identifier
- `max_length`: Maximum generation length
- `temperature`: Generation randomness
- `top_p`: Nucleus sampling parameter

### Embedding Settings

- `model_name`: Sentence Transformer model
- `embedding_dim`: Output dimension (auto)
- `device`: CPU or GPU

### Verification Settings

- `support_threshold`: Score for "supported" (default 0.7)
- `conflict_threshold`: Score for "conflicting" (default 0.5)
- `uncertainty_threshold`: Score for "unsupported" (default 0.4)

### Scoring Settings

- `supported_weight`: Weight for supported claims
- `partially_supported_weight`: Weight for partial claims
- `unsupported_weight`: Weight for unsupported claims
- `hallucination_penalty`: Penalty for hallucinations

### Highlighting Settings

- `high_risk_threshold`: Risk score threshold
- `medium_risk_threshold`: Medium risk threshold
- `color_scheme`: Color definitions

## 🚀 Usage Patterns

### Pattern 1: Basic Verification

```python
from main import HallucinationRadar
radar = HallucinationRadar()
radar.load_documents()
result = radar.verify_answer(question, answer)
```

### Pattern 2: Batch Processing

```python
qa_pairs = [...]
results = radar.batch_verify(qa_pairs)
```

### Pattern 3: Generate & Verify

```python
result = radar.generate_and_verify(question)
```

### Pattern 4: Custom Configuration

```python
radar = HallucinationRadar(config_path="custom_config.yaml")
```

### Pattern 5: Component-Level Access

```python
from claims.claim_extractor import ClaimExtractor
from verification.claim_verifier import ClaimVerifier

extractor = ClaimExtractor()
verifier = ClaimVerifier(vector_store)
```

## 📈 Output Structure

### Verification Result

```json
{
  "question": "...",
  "answer": "...",
  "truthfulness_score": 75.0,
  "report": {
    "score_category": "reliable",
    "claim_summary": {...},
    "risk_level": "low",
    "recommendations": [...]
  },
  "verification_results": [...],
  "highlighted_html": "...",
  "risk_summary": {...}
}
```

## 🔧 Customization Points

1. **LLM Model**: Change in config
2. **Evidence Sources**: Add documents to data/documents/
3. **Verification Thresholds**: Adjust in config
4. **Scoring Weights**: Configure in scoring section
5. **Embedding Model**: Choose different Sentence Transformers
6. **Highlighting Colors**: Customize color_scheme
7. **Claim Types**: Extend ClaimExtractor
8. **Verification Logic**: Override methods in ClaimVerifier

## 📋 Dependencies

Core dependencies in requirements.txt:

- torch
- transformers
- sentence-transformers
- faiss-cpu
- spacy
- wikipedia
- pdfplumber
- gradio
- numpy
- pandas
- pyyaml

## 🧪 Testing

### Unit Tests

- test_claim_extraction
- test_verification
- test_scoring
- test_highlighting

### Integration Tests

- test_full_pipeline
- test_batch_processing
- test_document_loading

### Evaluation

- FEVER benchmark evaluation
- Accuracy metrics
- F1 score calculation

## 📚 Code Organization

```
Source Code (17 files):
├── Main Scripts (2): app.py, main.py
├── Configuration (1): config/settings.yaml
├── Core Modules (10):
│   ├── llm/answer_generator.py
│   ├── claims/claim_extractor.py
│   ├── retireval/*.py (4 files)
│   ├── verification/claim_verifier.py
│   ├── scoring/truthfulness.py
│   ├── highlighting/highlighter.py
│   └── utils/text_utils.py
├── Evaluation (1): evaluation/fever_eval.py
└── Examples (1): QUICKSTART.py

Documentation (4 files):
├── README.md
├── BEST_PRACTICES.md
├── PROJECT_COMPLETION.md
└── INDEX.md (this file)
```

## 🔍 Finding What You Need

### For Users

- **Getting Started**: README.md → Usage section
- **Quick Examples**: QUICKSTART.py
- **Web Interface**: `python app.py`
- **Best Practices**: BEST_PRACTICES.md

### For Developers

- **Architecture**: PROJECT_COMPLETION.md
- **API Usage**: README.md → Python API section
- **Code Examples**: QUICKSTART.py
- **Component Details**: Individual module docstrings

### For Configuration

- **All Settings**: config/settings.yaml
- **Custom Config**: Duplicate settings.yaml and customize
- **Performance Tuning**: BEST_PRACTICES.md → Performance section

### For Troubleshooting

- **Common Issues**: BEST_PRACTICES.md → Troubleshooting
- **Error Handling**: main.py usage examples
- **Debugging**: Enable logging in BEST_PRACTICES.md

## 🎯 Next Steps

1. **Run Web Interface**: `python app.py`
2. **Add Documents**: Place files in `data/documents/`
3. **Test System**: Use examples in QUICKSTART.py
4. **Customize**: Edit config/settings.yaml
5. **Deploy**: Follow production checklist in BEST_PRACTICES.md

---

**Last Updated**: January 15, 2026  
**Version**: 1.0 - Complete  
**Status**: Production Ready ✅

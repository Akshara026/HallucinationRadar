# HallucinationRadar

**HallucinationRadar** is an advanced system that detects hallucinations in AI-generated content by verifying claims against trusted evidence sources.

Sometimes chatbots give confident answers that are wrong (**hallucinations**). HallucinationRadar acts like a **truth-checking layer** on top of AI responses using evidence-based verification.

---

## 🎯 What it does

When a user asks a question, the system:

1. **Generates an AI answer** (or accepts user-provided answer)
2. **Breaks the answer into factual statements (claims)** using NLP
3. **Searches trusted sources** (PDFs, documents, knowledge base) for evidence
4. **Evaluates each claim** using semantic similarity and textual overlap:
   - ✅ **Supported** by evidence
   - ⚠️ **Partially supported** / uncertain
   - ❌ **Unsupported** (potential hallucination)
   - 🚨 **Conflicting** evidence found
5. **Outputs a Truthfulness Score** (0–100)
6. **Highlights risky sentences** and provides **evidence citations**

---

## 📊 Output

HallucinationRadar provides comprehensive reports with:

- **Truthfulness Score**: 0–100% reliability rating
- **Score Category**: Highly Reliable / Reliable / Uncertain / Unreliable / Highly Unreliable
- **Claim-by-claim verification table** with evidence links
- **Risky/high-uncertainty sentence highlighting** with color coding
- **Evidence citations** from source documents
- **Risk assessment** and actionable recommendations

---

## ✨ Features

- **Evidence-Based Verification**: Uses your custom documents for fact-checking
- **Semantic Understanding**: Powered by state-of-the-art transformer models
- **Batch Processing**: Verify multiple questions in one go
- **Web Interface**: User-friendly Gradio interface
- **Detailed Reports**: Comprehensive claim breakdown and recommendations
- **Extensible Architecture**: Easy to add new evidence sources or verification methods

---

## 🚀 Installation

### Requirements

- Python 3.8+
- GPU recommended (CPU works but slower)

### Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd HallucinationRadar
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy model** (for NLP claim extraction)

   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Add your evidence documents**
   - Place PDF, TXT, or JSON files in `data/documents/`
   - These will be used for verification

---

## 📖 Usage

### Option 1: Web Interface (Easiest)

```bash
python app.py
```

Then open http://localhost:7860 in your browser.

Features:

- **Verify Answer**: Check any answer for hallucinations
- **Generate & Verify**: Let AI generate and verify in one step
- **Batch Verify**: Process multiple answers from CSV

### Option 2: Python API

```python
from main import HallucinationRadar

# Initialize
radar = HallucinationRadar()
radar.load_documents()

# Verify an answer
question = "What is photosynthesis?"
answer = "Photosynthesis is how plants make food using sunlight."

result = radar.verify_answer(question, answer)

print(f"Truthfulness Score: {result['truthfulness_score']:.1f}")
print(f"Category: {result['report']['score_category']}")
```

### Option 3: Generate & Verify

```python
# Generate answer and verify in one pipeline
result = radar.generate_and_verify("What is the capital of France?")

print(f"Generated Answer: {result['answer']}")
print(f"Truthfulness Score: {result['truthfulness_score']:.1f}")
```

---

## 📁 Project Structure

```
HallucinationRadar/
├── app.py                      # Gradio web interface
├── main.py                     # Main orchestration pipeline
├── requirements.txt            # Python dependencies
├── config/
│   └── settings.yaml          # Configuration file
├── llm/
│   └── answer_generator.py    # LLM answer generation
├── claims/
│   └── claim_extractor.py     # Extract claims from text
├── retireval/                  # (Note: folder name typo, kept as-is)
│   ├── doc_loader.py          # Load documents from files
│   ├── embedder.py            # Create text embeddings
│   ├── vector_store.py        # Semantic search
│   └── web_search.py          # Web search integration
├── verification/
│   └── claim_verifier.py      # Verify claims against evidence
├── scoring/
│   └── truthfulness.py        # Calculate truthfulness scores
├── highlighting/
│   └── highlighter.py         # Highlight risky claims
├── evaluation/
│   └── fever_eval.py          # Evaluation metrics
├── utils/
│   └── text_utils.py          # Text processing utilities
└── data/
    └── documents/             # Add evidence documents here
```

---

## ⚙️ Configuration

Edit `config/settings.yaml` to customize:

```yaml
# LLM Settings
llm:
  model_name: "gpt2" # Change to larger models if desired
  max_length: 256
  temperature: 0.7

# Verification thresholds
verification:
  support_threshold: 0.7 # Score needed for "supported"
  conflict_threshold: 0.5
  uncertainty_threshold: 0.4

# Scoring weights
scoring:
  supported_weight: 1.0
  partially_supported_weight: 0.5
  unsupported_weight: 0.0

# Highlighting colors
highlighting:
  high_risk_threshold: 0.3
  medium_risk_threshold: 0.6
```

---

## 📊 Score Interpretation

| Score Range | Category          | Meaning                               |
| ----------- | ----------------- | ------------------------------------- |
| 80-100      | Highly Reliable   | Strong evidence supporting the answer |
| 60-79       | Reliable          | Most claims verified, minor concerns  |
| 40-59       | Uncertain         | Mixed evidence, proceed with caution  |
| 20-39       | Unreliable        | Multiple unverified claims            |
| 0-19        | Highly Unreliable | Mostly unsupported claims             |

---

## 🧠 How It Works

### 1. Claim Extraction

- Uses spaCy NLP to identify factual statements
- Extracts entities, verbs, and relationships
- Classifies claims by type (factual, numerical, temporal, etc.)

### 2. Evidence Retrieval

- Creates embeddings for all documents using Sentence Transformers
- Performs semantic search for relevant documents
- Returns top-K most similar documents

### 3. Claim Verification

- Compares each claim against retrieved evidence
- Uses semantic similarity + textual overlap
- Assigns support/conflict status based on thresholds

### 4. Truthfulness Scoring

- Weights claims by verification status
- Combines scores using configured weights
- Normalizes to 0-100 scale

### 5. Risk Highlighting

- Maps verification results to sentences
- Assigns risk levels (high/medium/low)
- Creates HTML visualization

---

## 🔧 Extending the System

### Add Custom LLM

```python
class AnswerGenerator:
    def __init__(self, model_name="your-model"):
        # Load your custom model
        self.model = load_model(model_name)
```

### Add Evidence Sources

```python
loader = DocumentLoader()

# Load from multiple sources
documents = loader.load_documents()  # From data/documents/
# Add custom sources
documents.extend(load_wikipedia_articles(topics))
documents.extend(load_news_articles(keywords))

vector_store.add_documents(documents)
```

### Custom Verification Logic

```python
verifier = ClaimVerifier(vector_store)
# Override verification thresholds
verifier.support_threshold = 0.8
verifier.conflict_threshold = 0.4
```

---

## ⚡ Performance Tips

1. **GPU Acceleration**: Set device to 'cuda' in config for faster embeddings
2. **Smaller Models**: Use 'all-MiniLM-L6-v2' for CPU efficiency
3. **Batch Processing**: Use batch_verify for multiple answers
4. **Caching**: Enable embedding cache in config

---

## 🐛 Troubleshooting

**Issue**: "No documents loaded"

- **Solution**: Add documents to `data/documents/` folder (PDF, TXT, or JSON)

**Issue**: Model download fails

- **Solution**: Check internet connection, or download separately
  ```bash
  python -m spacy download en_core_web_sm
  ```

**Issue**: GPU out of memory

- **Solution**: Set device to 'cpu' in config, or use smaller model

---

## 📚 Dependencies

- **torch**: Deep learning framework
- **transformers**: Pre-trained LLM models
- **sentence-transformers**: Embedding models
- **spacy**: NLP processing
- **faiss-cpu**: Vector similarity search
- **pdfplumber**: PDF reading
- **gradio**: Web interface
- **numpy, pandas**: Data processing
- **pyyaml**: Configuration management

---

## 📄 License

See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Support for more LLM models (GPT-4, Claude, Llama)
- [ ] Web search integration (Google, Bing APIs)
- [ ] More evaluation benchmarks (SQuAD, FEVER)
- [ ] Multi-language support
- [ ] Real-time fact-checking
- [ ] Integration with fact-checking APIs

---

## 📞 Support

For issues or questions, please open an issue on the repository.

---

## 🙏 Acknowledgments

- FEVER dataset for evaluation framework
- Hugging Face for transformer models
- Sentence-Transformers for embedding models
- spaCy for NLP toolkit

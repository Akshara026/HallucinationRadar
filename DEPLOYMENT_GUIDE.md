# 🚀 HallucinationRadar - Deployment Guide

## ✅ Deployment Status: COMPLETE

The HallucinationRadar project has been successfully deployed and is ready for use!

---

## 📋 Deployment Checklist

### Environment Setup ✅

- [x] Python 3.12.2 verified
- [x] All dependencies installed
- [x] spaCy en_core_web_sm model downloaded
- [x] Data directories created
- [x] Configuration verified
- [x] All core imports tested
- [x] Web UI module tested
- [x] Main API module tested
- [x] Launch scripts created

### Project Structure ✅

- [x] 17 Python modules
- [x] 1 Configuration file
- [x] 6 Documentation files
- [x] Core modules (11 files)
- [x] Entry points (3 files)

---

## 🚀 Getting Started

### Option 1: Launch Web Interface (Easiest)

**Windows:**

```bash
# Double-click launch_web.bat or:
python app.py
```

**Linux/Mac:**

```bash
python app.py
```

Then open your browser to: **http://localhost:7860**

### Option 2: Use Python API

```python
from main import HallucinationRadar

# Initialize
radar = HallucinationRadar()

# Load your documents
radar.load_documents()

# Verify an answer
result = radar.verify_answer(
    question="What is the capital of France?",
    answer="Paris is the capital of France."
)

# Check results
print(f"Score: {result['truthfulness_score']:.1f}")
print(f"Category: {result['report']['score_category']}")
```

### Option 3: Run Examples

```bash
python QUICKSTART.py
```

---

## 📁 Project Structure

```
HallucinationRadar/
├── app.py                      # Web interface (Gradio)
├── main.py                     # Python API
├── deploy.py                   # Deployment script
├── launch_web.bat              # Windows launcher
│
├── Core Modules/
│   ├── llm/answer_generator.py
│   ├── claims/claim_extractor.py
│   ├── retireval/
│   │   ├── doc_loader.py
│   │   ├── embedder.py
│   │   ├── vector_store.py
│   │   └── web_search.py
│   ├── verification/claim_verifier.py
│   ├── scoring/truthfulness.py
│   ├── highlighting/highlighter.py
│   ├── evaluation/fever_eval.py
│   └── utils/text_utils.py
│
├── Configuration/
│   └── config/settings.yaml
│
├── Documentation/
│   ├── README.md
│   ├── BEST_PRACTICES.md
│   ├── PROJECT_COMPLETION.md
│   ├── INDEX.md
│   └── FINAL_REPORT.md
│
├── Data/
│   ├── documents/              # ← Add your evidence documents here
│   └── vector_store/           # Auto-generated embeddings
│
└── Requirements/
    └── requirements.txt
```

---

## 📂 Adding Evidence Documents

### Supported Formats

- **PDF Files** (.pdf)
- **Text Files** (.txt)
- **JSON Files** (.json)

### Steps:

1. Prepare your documents
2. Place them in: `data/documents/`
3. Restart the application (if running)
4. Documents will be automatically indexed

### Example Structure:

```
data/documents/
├── wikipedia/
│   ├── science.txt
│   ├── history.txt
│   └── geography.txt
├── academic_papers/
│   ├── paper1.pdf
│   └── paper2.pdf
└── custom_knowledge/
    ├── facts.json
    └── definitions.txt
```

---

## ⚙️ Configuration

### Quick Configuration

Edit `config/settings.yaml` to customize:

**For Faster Processing (CPU):**

```yaml
llm:
  model_name: "gpt2"
  max_length: 128

embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"
```

**For Better Accuracy (if GPU available):**

```yaml
llm:
  model_name: "facebook/opt-350m"
  max_length: 256

embedding:
  model_name: "sentence-transformers/all-mpnet-base-v2"
  device: "cuda"
```

**Adjust Verification Sensitivity:**

```yaml
verification:
  support_threshold: 0.7 # Higher = stricter
  uncertainty_threshold: 0.4 # Lower = more lenient
```

---

## 🎯 Web Interface Guide

### Tab 1: Verify Answer

- Enter a question
- Provide an answer to verify
- Get truthfulness score, claim breakdown, and risk highlights

### Tab 2: Generate & Verify

- Enter a question
- System generates an answer
- Returns verification results

### Tab 3: Batch Verify

- Upload CSV with Question, Answer columns
- Process multiple items
- Download results CSV

### Tab 4: About

- Project information
- Feature overview
- Usage guidelines

---

## 📊 Output Interpretation

### Truthfulness Score

- **80-100**: 🟢 Highly Reliable
- **60-79**: 🟡 Reliable
- **40-59**: 🟠 Uncertain
- **20-39**: 🔴 Unreliable
- **0-19**: 🔴🔴 Highly Unreliable

### Claim Status

- ✅ **Supported**: Strong evidence
- ⚠️ **Partially Supported**: Some evidence
- ❌ **Unsupported**: No evidence
- 🚨 **Conflicting**: Contradicting evidence

---

## 🔧 Troubleshooting

### Issue: "No documents loaded"

**Solution**: Add documents to `data/documents/` and restart

### Issue: Out of memory

**Solution**: In config/settings.yaml, set device to "cpu" and use smaller models

### Issue: Slow inference

**Solution**:

- Enable GPU (if available)
- Use smaller embedding model
- Reduce batch size

### Issue: Module import errors

**Solution**:

```bash
pip install -r requirements.txt --force-reinstall
python -m spacy download en_core_web_sm
```

### Issue: Port already in use

**Solution**: Change port in app.py or kill existing process:

```bash
# Find process on port 7860
lsof -i :7860  # Mac/Linux
netstat -ano | findstr :7860  # Windows
```

---

## 📈 Performance Optimization

### CPU Mode (Recommended for most users)

```python
radar = HallucinationRadar()
# Automatically uses CPU if GPU not available
```

### GPU Mode (for faster processing)

```yaml
embedding:
  device: "cuda"
llm:
  model_name: "facebook/opt-1.3b"
```

### Batch Processing

```python
qa_pairs = [
    {'question': 'Q1', 'answer': 'A1'},
    {'question': 'Q2', 'answer': 'A2'},
    # ...
]
results = radar.batch_verify(qa_pairs)
```

---

## 🔒 Security Considerations

- ✅ Keep sensitive documents in secure folder
- ✅ Don't share results containing PII
- ✅ Use in controlled environments
- ✅ Validate document sources

---

## 📊 Monitoring

### Enable Logging

In Python code:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Check Logs

Logs are stored in: `logs/hallucination_radar.log`

### Monitor Metrics

```python
from main import HallucinationRadar

radar = HallucinationRadar()
radar.load_documents()

# Track scores
results = radar.batch_verify(qa_pairs)
avg_score = sum(r['truthfulness_score'] for r in results) / len(results)
print(f"Average Score: {avg_score:.1f}")
```

---

## 🔄 Maintenance

### Regular Tasks

1. **Update Documents**: Add new evidence regularly
2. **Check Performance**: Monitor inference times
3. **Review Results**: Validate accuracy
4. **Update Models**: Periodically update embedding models

### Backup Important Data

```bash
# Backup vector store
cp -r data/vector_store/ data/vector_store.backup/

# Backup configuration
cp config/settings.yaml config/settings.yaml.backup
```

---

## 🚀 Deployment Environments

### Local Development

```bash
python app.py
# Access: http://localhost:7860
```

### Docker (Optional - for production)

Create a `Dockerfile`:

```dockerfile
FROM python:3.12

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && \
    python -m spacy download en_core_web_sm

COPY . .

EXPOSE 7860
CMD ["python", "app.py"]
```

Build and run:

```bash
docker build -t hallucination-radar .
docker run -p 7860:7860 hallucination-radar
```

### Cloud Deployment (Heroku Example)

1. Create `Procfile`: `web: python app.py`
2. Create `runtime.txt`: `python-3.12.2`
3. Deploy: `git push heroku main`

---

## 📞 Support & Resources

### Documentation

- 📖 **README.md** - Full project overview
- 📖 **BEST_PRACTICES.md** - Usage patterns
- 📖 **QUICKSTART.py** - Code examples
- 📖 **INDEX.md** - Project organization

### Getting Help

1. Check documentation files
2. Review example code
3. Check configuration options
4. Review troubleshooting section

---

## ✨ What's Included

### Features

✅ LLM answer generation  
✅ NLP claim extraction  
✅ Multi-format document loading  
✅ Semantic search  
✅ Evidence-based verification  
✅ Truthfulness scoring  
✅ Risk highlighting  
✅ Batch processing  
✅ Web interface  
✅ Python API

### Tools

✅ Gradio web UI  
✅ Python API  
✅ Command-line examples  
✅ Deployment script  
✅ Launch scripts  
✅ Comprehensive documentation

---

## 🎯 Next Steps

1. **Add Documents**: Place files in `data/documents/`
2. **Configure Settings**: Edit `config/settings.yaml` if needed
3. **Start Using**: Run `python app.py`
4. **Integrate**: Use in your workflows

---

## 📝 Deployment Verification

To verify deployment is complete:

```bash
# Check all components
python verify_project.py

# Test imports
python -c "from main import HallucinationRadar; print('✓ Ready!')"

# Run examples
python QUICKSTART.py
```

---

## 🎉 You're Ready!

**Deployment Successful!** ✅

HallucinationRadar is now deployed and ready for production use.

- 🌐 **Web UI**: `python app.py` → http://localhost:7860
- 🐍 **Python API**: `from main import HallucinationRadar`
- 📚 **Documentation**: See README.md and guides
- 💡 **Examples**: Run QUICKSTART.py

Enjoy fact-checking with HallucinationRadar! 🚀

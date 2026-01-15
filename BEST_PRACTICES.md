# Best Practices Guide for HallucinationRadar

## 🎯 Best Practices for Using HallucinationRadar

### 1. Document Management

**Do:**

- ✅ Organize documents by topic in `data/documents/`
- ✅ Use clear, descriptive filenames
- ✅ Regularly update document collection
- ✅ Include diverse sources for better verification

**Don't:**

- ❌ Mix unrelated documents in same folder
- ❌ Use very old or outdated documents
- ❌ Keep documents with confidential information
- ❌ Store extremely large files (>100MB)

**Example Structure:**

```
data/documents/
├── wikipedia_articles/
│   ├── science.txt
│   ├── history.txt
│   └── geography.txt
├── technical_docs/
│   ├── python_guide.pdf
│   └── ai_fundamentals.pdf
└── reference/
    ├── facts.json
    └── definitions.txt
```

### 2. Configuration Management

**LLM Configuration:**

```yaml
# For accuracy (slower)
llm:
  model_name: "facebook/opt-350m"
  temperature: 0.5

# For speed (faster, less accurate)
llm:
  model_name: "gpt2"
  temperature: 0.3
```

**Verification Thresholds:**

```yaml
# Strict verification
verification:
  support_threshold: 0.8
  uncertainty_threshold: 0.5

# Lenient verification
verification:
  support_threshold: 0.6
  uncertainty_threshold: 0.3
```

### 3. Error Handling

**Recommended Pattern:**

```python
from main import HallucinationRadar
import logging

logging.basicConfig(level=logging.INFO)

try:
    radar = HallucinationRadar()
    if not radar.load_documents():
        print("Warning: No documents loaded")

    result = radar.verify_answer(question, answer)

    if result.get('error'):
        print(f"Verification error: {result['error']}")
    else:
        print(f"Score: {result['truthfulness_score']}")

except FileNotFoundError:
    print("Config file not found")
except Exception as e:
    logging.error(f"Unexpected error: {e}")
```

### 4. Performance Optimization

**For CPU Systems:**

```yaml
embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"

llm:
  model_name: "gpt2"
  max_length: 128
```

**For GPU Systems:**

```yaml
embedding:
  model_name: "sentence-transformers/all-mpnet-base-v2"
  device: "cuda"

llm:
  model_name: "facebook/opt-350m"
  max_length: 512
```

**Batch Processing Best Practices:**

```python
# Good: Process in batches
qa_pairs = [...]
results = radar.batch_verify(qa_pairs)

# Avoid: Processing one by one in a loop
# for qa in qa_pairs:
#     radar.verify_answer(qa['question'], qa['answer'])
```

### 5. Result Interpretation

**Check Score First:**

```python
score = result['truthfulness_score']
if score >= 80:
    print("✓ Highly reliable")
elif score >= 60:
    print("⚠ Generally reliable")
elif score >= 40:
    print("⚠ Mixed evidence")
else:
    print("✗ Potentially unreliable")
```

**Always Review Unsupported Claims:**

```python
unsupported = result['report']['claim_breakdown']['unsupported_claims']
if unsupported:
    print("Review these claims independently:")
    for claim in unsupported:
        print(f"  - {claim}")
```

**Use Recommendations:**

```python
recommendations = result['report']['recommendations']
for rec in recommendations:
    print(f"→ {rec}")
```

### 6. Integration Patterns

**Pattern 1: Simple Verification**

```python
radar = HallucinationRadar()
radar.load_documents()
result = radar.verify_answer(q, a)
```

**Pattern 2: With Error Handling**

```python
try:
    radar = HallucinationRadar()
    if radar.load_documents():
        result = radar.verify_answer(q, a)
        return result
    else:
        return {"error": "No documents loaded"}
except Exception as e:
    return {"error": str(e)}
```

**Pattern 3: Web Service**

```python
from fastapi import FastAPI
from main import HallucinationRadar

app = FastAPI()
radar = HallucinationRadar()
radar.load_documents()

@app.post("/verify")
async def verify(question: str, answer: str):
    return radar.verify_answer(question, answer)
```

**Pattern 4: Streaming Results**

```python
def verify_stream(qa_pairs):
    for qa in qa_pairs:
        result = radar.verify_answer(qa['q'], qa['a'])
        yield result
```

### 7. Common Pitfalls to Avoid

**❌ Pitfall 1: No Documents Loaded**

```python
# Wrong
radar = HallucinationRadar()
result = radar.verify_answer(q, a)  # Will have low scores

# Right
radar = HallucinationRadar()
radar.load_documents()  # Load first!
result = radar.verify_answer(q, a)
```

**❌ Pitfall 2: Ignoring Score Categories**

```python
# Wrong
if result['truthfulness_score'] > 50:
    trust_answer = True

# Right
category = result['report']['score_category']
if category in ['highly_reliable', 'reliable']:
    trust_answer = True
```

**❌ Pitfall 3: Not Reviewing Edge Cases**

```python
# Wrong
result = radar.verify_answer(q, a)
if result['truthfulness_score'] > 70:
    use_answer = True

# Right
if result['report']['claim_summary']['conflicting'] > 0:
    review_manually = True
elif result['truthfulness_score'] > 70:
    use_answer = True
```

**❌ Pitfall 4: Stale Models**

```python
# Update periodically
# Use newer embedding models for better accuracy
embedding:
  model_name: "sentence-transformers/all-mpnet-base-v2"  # Updated
```

### 8. Monitoring & Logging

**Setup Logging:**

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hallucination_radar.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

**Track Metrics:**

```python
def analyze_batch(results):
    avg_score = sum(r['truthfulness_score'] for r in results) / len(results)
    high_confidence = sum(1 for r in results if r['truthfulness_score'] > 75)

    print(f"Average Score: {avg_score:.1f}")
    print(f"High Confidence: {high_confidence}/{len(results)}")
```

### 9. Security Considerations

**Document Handling:**

```python
# Sanitize sensitive information
def sanitize_documents():
    # Remove PII, credentials, sensitive data
    # Consider encryption for sensitive docs
    pass

# Validate document sources
def validate_source(doc_path):
    # Check file integrity
    # Verify source authenticity
    pass
```

**Result Sharing:**

```python
# Be cautious with sensitive results
def export_result(result, include_evidence=False):
    safe_result = {
        'truthfulness_score': result['truthfulness_score'],
        'score_category': result['report']['score_category'],
    }

    if include_evidence:
        safe_result['recommendations'] = result['report']['recommendations']

    return safe_result
```

### 10. Testing & Validation

**Unit Testing Pattern:**

```python
def test_claim_extraction():
    text = "The Earth is round and orbits the sun."
    extractor = ClaimExtractor()
    claims = extractor.extract_claims(text)

    assert len(claims) > 0
    assert any('Earth' in c['claim'] for c in claims)

def test_verification():
    radar = HallucinationRadar()
    radar.load_documents()

    result = radar.verify_answer("Q", "A")

    assert 'truthfulness_score' in result
    assert 0 <= result['truthfulness_score'] <= 100
```

**Integration Testing:**

```python
def test_full_pipeline():
    qa_pairs = [
        {'question': 'Q1', 'answer': 'A1'},
        {'question': 'Q2', 'answer': 'A2'},
    ]

    results = radar.batch_verify(qa_pairs)

    assert len(results) == len(qa_pairs)
    for result in results:
        assert 'truthfulness_score' in result
        assert 'report' in result
```

---

## 📋 Configuration Checklist

- [ ] Add documents to `data/documents/`
- [ ] Adjust thresholds based on use case
- [ ] Choose appropriate LLM model
- [ ] Set device (cuda/cpu)
- [ ] Configure logging
- [ ] Test with sample data
- [ ] Review security settings
- [ ] Set up monitoring

---

## 🚀 Production Deployment Checklist

- [ ] Load test the system
- [ ] Set up logging and monitoring
- [ ] Configure authentication/authorization
- [ ] Implement rate limiting
- [ ] Set up backup system
- [ ] Document API usage
- [ ] Create runbooks
- [ ] Test error scenarios
- [ ] Performance tune configurations
- [ ] Plan for updates

---

## 📞 Troubleshooting Guide

**Issue: Low verification scores for correct claims**

- Solution: Add more relevant documents to knowledge base

**Issue: High memory usage**

- Solution: Use smaller embedding model, reduce batch size

**Issue: Slow inference**

- Solution: Enable GPU, use smaller LLM model

**Issue: Inconsistent results**

- Solution: Check document quality, verify thresholds

---

**Remember**: HallucinationRadar is a tool to assist verification, not replace human judgment. Always review critical information independently!

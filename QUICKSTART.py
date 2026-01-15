"""
QUICK START GUIDE - HallucinationRadar
=====================================

Get started in 5 minutes!
"""

# Step 1: Install dependencies (already done!)
# pip install -r requirements.txt

# Step 2: Download spaCy model (if not already done)
# python -m spacy download en_core_web_sm

# Step 3: Add evidence documents
# Place your PDF/TXT/JSON files in: data/documents/

# Step 4: Run the web interface
# python app.py
# Then open: http://localhost:7860

# ====== EXAMPLE 1: Simple Verification ======

from main import HallucinationRadar

# Initialize
print("Initializing HallucinationRadar...")
radar = HallucinationRadar()

# Load your documents
print("Loading documents...")
radar.load_documents()

# Verify an answer
question = "What is the Earth's largest moon?"
answer = "The Moon is Earth's largest satellite and orbits the planet every 27 days."

print(f"\nQuestion: {question}")
print(f"Answer: {answer}")

result = radar.verify_answer(question, answer)

print(f"\n✓ Truthfulness Score: {result['truthfulness_score']:.1f}/100")
print(f"✓ Category: {result['report']['score_category']}")
print(f"✓ Claims Found: {result['report']['claim_summary']['total_claims']}")

print("\nSupported Claims:")
for claim in result['report']['claim_breakdown']['supported_claims'][:3]:
    print(f"  ✓ {claim}")

print("\nUnsupported Claims:")
for claim in result['report']['claim_breakdown']['unsupported_claims'][:3]:
    print(f"  ✗ {claim}")


# ====== EXAMPLE 2: Batch Verification ======

qa_pairs = [
    {
        'question': 'What is photosynthesis?',
        'answer': 'Photosynthesis is how plants convert sunlight into chemical energy.'
    },
    {
        'question': 'How many continents are there?',
        'answer': 'There are 7 continents on Earth.'
    }
]

print("\n\nBatch Verification Results:")
results = radar.batch_verify(qa_pairs)

for result in results:
    print(f"\nQ: {result['question']}")
    print(f"Score: {result['truthfulness_score']:.1f} - {result['report']['score_category']}")


# ====== EXAMPLE 3: Generate & Verify ======

question = "Who was the first President of the United States?"

print(f"\n\nGenerating answer for: {question}")
result = radar.generate_and_verify(question)

print(f"Generated: {result['answer']}")
print(f"Truthfulness: {result['truthfulness_score']:.1f}")


# ====== WORKING WITH RESULTS ======

# Access detailed verification results
verification_results = result['verification_results']

print("\n\nDetailed Verification Results:")
for ver_result in verification_results[:3]:
    print(f"\nClaim: {ver_result['claim']}")
    print(f"Status: {ver_result['status']}")
    print(f"Confidence: {ver_result['confidence']:.2%}")
    
    # Show evidence
    if ver_result['evidence']:
        print(f"Evidence: {ver_result['evidence'][0]['title']}")


# ====== USEFUL FUNCTIONS ======

# Get score description
scorer = radar.truthfulness_scorer
description = scorer.get_score_description(75)
print(f"\nScore 75 means: {description}")

# Get risk level
risk_level = scorer._calculate_risk_level(75, [], [])
print(f"Risk level: {risk_level}")

# Generate full report
report = result['report']
print(f"\nRecommendations:")
for rec in report['recommendations']:
    print(f"• {rec}")


# ====== CONFIGURATION ====== 

# Change verification thresholds in config/settings.yaml:
# verification:
#   support_threshold: 0.7        # Confidence needed for "supported"
#   conflict_threshold: 0.5
#   uncertainty_threshold: 0.4

# Adjust LLM parameters:
# llm:
#   model_name: "gpt2"            # Try: "facebook/opt-350m", "distilgpt2"
#   temperature: 0.7              # Lower = more focused, Higher = more creative
#   max_length: 256


# ====== ADDING CUSTOM DOCUMENTS ====== 

# You can add documents in three ways:

# 1. Place files in data/documents/
#    - PDF files (.pdf)
#    - Text files (.txt)  
#    - JSON files (.json)

# 2. Programmatically add documents
from retireval.doc_loader import DocumentLoader

loader = DocumentLoader()
# Documents are auto-loaded from data/documents/

# 3. Add custom documents programmatically
custom_docs = [
    {
        'id': 'doc1',
        'title': 'Solar System Facts',
        'content': 'The solar system has 8 planets. Mercury is the closest to the sun.'
    },
    {
        'id': 'doc2',
        'title': 'Moon Facts',
        'content': 'The Moon orbits Earth every 27.3 days.'
    }
]

# Add to vector store
# radar.vector_store.add_documents(custom_docs)


# ====== COMMON ISSUES & SOLUTIONS ======

# Issue: "No documents loaded"
# Solution: Add PDFs or TXT files to data/documents/

# Issue: Out of memory on GPU
# Solution: In config/settings.yaml, set device: cpu

# Issue: Slow inference
# Solution: Use smaller embedding model or enable GPU

# Issue: Spacy model not found
# Solution: python -m spacy download en_core_web_sm


print("\n✓ Quick Start Complete!")
print("For more features, launch the web interface: python app.py")

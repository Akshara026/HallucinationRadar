"""Gradio web interface for HallucinationRadar."""

import gradio as gr
import logging
from main import HallucinationRadar
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize radar
logger.info("Initializing HallucinationRadar web interface...")
radar = HallucinationRadar()

# Load documents
radar.load_documents()

# Define interface functions
def verify_answer_interface(question: str, answer: str) -> tuple:
    """Verify an answer through the web interface."""
    try:
        result = radar.verify_answer(question, answer)
        
        # Format output
        score = result['truthfulness_score']
        category = result['report']['score_category']
        description = result['report']['description']
        
        # Create claims table
        claims_md = "### Verification Results\n\n"
        claims_md += "| Claim | Status | Confidence |\n"
        claims_md += "|-------|--------|------------|\n"
        
        for ver_result in result['verification_results'][:10]:
            claim = ver_result['claim'][:50]
            status = ver_result['status']
            confidence = f"{ver_result['confidence']:.2%}"
            claims_md += f"| {claim}... | {status} | {confidence} |\n"
        
        # Create summary
        summary_md = f"""
### Truthfulness Report

**Score:** {score:.1f}/100

**Category:** {category}

**Description:** {description}

**Claim Breakdown:**
- Supported: {result['report']['claim_summary']['supported']}
- Partially Supported: {result['report']['claim_summary']['partially_supported']}
- Unsupported: {result['report']['claim_summary']['unsupported']}
- Conflicting: {result['report']['claim_summary']['conflicting']}

**Risk Level:** {result['report']['risk_level'].upper()}

**Recommendations:**
"""
        for rec in result['report']['recommendations']:
            summary_md += f"- {rec}\n"
        
        return score, category, summary_md, claims_md, result['highlighted_html']
    
    except Exception as e:
        logger.error(f"Error in verification: {e}")
        error_msg = f"**Error:** {str(e)}"
        return 0, "error", error_msg, "", ""


def generate_and_verify_interface(question: str) -> tuple:
    """Generate and verify an answer."""
    try:
        result = radar.generate_and_verify(question)
        
        answer = result['answer']
        score = result['truthfulness_score']
        category = result['report']['score_category']
        description = result['report']['description']
        
        # Create summary
        summary_md = f"""
### Generated Answer

**Answer:** {answer}

### Truthfulness Report

**Score:** {score:.1f}/100

**Category:** {category}

**Description:** {description}

**Claim Breakdown:**
- Supported: {result['report']['claim_summary']['supported']}
- Partially Supported: {result['report']['claim_summary']['partially_supported']}
- Unsupported: {result['report']['claim_summary']['unsupported']}
- Conflicting: {result['report']['claim_summary']['conflicting']}

**Recommendations:**
"""
        for rec in result['report']['recommendations']:
            summary_md += f"- {rec}\n"
        
        return answer, score, category, summary_md
    
    except Exception as e:
        logger.error(f"Error in generation: {e}")
        return f"Error: {str(e)}", 0, "error", f"**Error:** {str(e)}"


def batch_verify_interface(csv_input: str) -> str:
    """Verify multiple questions and answers from CSV."""
    try:
        lines = csv_input.strip().split('\n')
        
        qa_pairs = []
        for line in lines[1:]:  # Skip header
            parts = line.split(',', 1)
            if len(parts) == 2:
                qa_pairs.append({
                    'question': parts[0].strip(),
                    'answer': parts[1].strip()
                })
        
        results = radar.batch_verify(qa_pairs)
        
        # Create output CSV
        output = "Question,Answer,Truthfulness Score,Category\n"
        for result in results:
            q = result['question'].replace(',', ';')
            a = result['answer'][:50].replace(',', ';')
            score = result['truthfulness_score']
            category = result['report']['score_category']
            output += f"\"{q}\",\"{a}...\",{score:.1f},{category}\n"
        
        return output
    
    except Exception as e:
        logger.error(f"Error in batch verification: {e}")
        return f"Error: {str(e)}"


# Create Gradio interface with tabs
with gr.Blocks(title="HallucinationRadar", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎯 HallucinationRadar")
    gr.Markdown("Detect hallucinations in AI-generated answers with evidence-based verification")
    
    with gr.Tabs():
        # Tab 1: Verify Answer
        with gr.TabItem("Verify Answer"):
            gr.Markdown("### Verify an existing answer for hallucinations")
            
            with gr.Row():
                with gr.Column():
                    question_input = gr.Textbox(
                        label="Question",
                        placeholder="Enter your question...",
                        lines=2
                    )
                    answer_input = gr.Textbox(
                        label="Answer to Verify",
                        placeholder="Enter the answer you want to verify...",
                        lines=4
                    )
                    verify_btn = gr.Button("Verify Answer", variant="primary")
                
                with gr.Column():
                    score_output = gr.Number(label="Truthfulness Score (0-100)")
                    category_output = gr.Textbox(label="Category")
            
            summary_output = gr.Markdown(label="Report")
            claims_output = gr.Markdown(label="Claims Verification")
            html_output = gr.HTML(label="Highlighted Answer")
            
            verify_btn.click(
                verify_answer_interface,
                inputs=[question_input, answer_input],
                outputs=[score_output, category_output, summary_output, claims_output, html_output]
            )
        
        # Tab 2: Generate & Verify
        with gr.TabItem("Generate & Verify"):
            gr.Markdown("### Generate an answer and verify it in one step")
            
            with gr.Row():
                with gr.Column():
                    question_gen = gr.Textbox(
                        label="Question",
                        placeholder="Enter your question...",
                        lines=2
                    )
                    gen_verify_btn = gr.Button("Generate & Verify", variant="primary")
                
                with gr.Column():
                    answer_output = gr.Textbox(label="Generated Answer", lines=3)
                    score_gen = gr.Number(label="Truthfulness Score")
                    category_gen = gr.Textbox(label="Category")
            
            summary_gen = gr.Markdown(label="Report")
            
            gen_verify_btn.click(
                generate_and_verify_interface,
                inputs=[question_gen],
                outputs=[answer_output, score_gen, category_gen, summary_gen]
            )
        
        # Tab 3: Batch Verification
        with gr.TabItem("Batch Verify"):
            gr.Markdown("### Verify multiple question-answer pairs")
            gr.Markdown("Upload CSV with columns: Question, Answer")
            
            with gr.Row():
                csv_input = gr.Textbox(
                    label="CSV Input (Question, Answer)",
                    placeholder="Question 1,Answer 1\nQuestion 2,Answer 2\n...",
                    lines=10,
                    max_lines=100
                )
                batch_btn = gr.Button("Process Batch", variant="primary")
            
            csv_output = gr.Textbox(
                label="CSV Output",
                lines=10,
                max_lines=100
            )
            
            batch_btn.click(
                batch_verify_interface,
                inputs=[csv_input],
                outputs=[csv_output]
            )
        
        # Tab 4: About
        with gr.TabItem("About"):
            gr.Markdown("""
            ## About HallucinationRadar
            
            HallucinationRadar is an advanced AI hallucination detection system that:
            
            1. **Extracts Claims** - Identifies factual statements from answers
            2. **Searches Evidence** - Finds relevant documents from your knowledge base
            3. **Verifies Claims** - Compares claims against evidence using semantic similarity
            4. **Scores Truthfulness** - Calculates an overall reliability score (0-100)
            5. **Highlights Risks** - Marks uncertain or risky statements
            
            ### How to Use
            
            - **Verify Answer**: Provide a question and answer to check for hallucinations
            - **Generate & Verify**: Let AI generate an answer and verify it automatically
            - **Batch Verify**: Check multiple answers at once
            
            ### Features
            
            - **Evidence-Based**: Uses your documents for verification
            - **Semantic Understanding**: Leverages transformer models for deep comprehension
            - **Detailed Reports**: Get claims breakdown and recommendations
            - **Risk Highlighting**: Visual indicators for uncertain statements
            
            ### Thresholds
            
            - **Highly Reliable**: Score ≥ 80
            - **Reliable**: Score 60-79
            - **Uncertain**: Score 40-59
            - **Unreliable**: Score 20-39
            - **Highly Unreliable**: Score < 20
            """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

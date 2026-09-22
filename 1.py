import streamlit as st
from graph import run_pipeline

st.title("HallucinationRadar")
text = st.text_area("Paste text to fact-check")
if st.button("Check"):
    with st.spinner("Analyzing..."):
        result = run_pipeline(external_answer=text)
    st.metric("Hallucination Score", result["score"]["overall_score"])
    # render claims, verdicts, etc. as st.write() or st.dataframe()
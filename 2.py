python
import streamlit as st
from graph import run_pipeline

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="HallucinationRadar",
    page_icon="🔎",
    layout="wide",
)

# -----------------------------
# Header
# -----------------------------
st.title("🔎 HallucinationRadar")
st.markdown(
    "Detect potentially **hallucinated claims**, evaluate their confidence, "
    "and inspect supporting evidence."
)

st.divider()

# -----------------------------
# Input
# -----------------------------
st.subheader("📝 Text to fact-check")

text = st.text_area(
    "Paste an answer, article, or response",
    height=220,
    placeholder=(
        "Example:\n\n"
        "The Eiffel Tower was completed in 1899 and is located in London."
    ),
    label_visibility="collapsed",
)

check_button = st.button(
    "🔍 Check for Hallucinations",
    type="primary",
    use_container_width=True,
)

# -----------------------------
# Pipeline
# -----------------------------
if check_button:

    if not text.strip():
        st.warning("⚠️ Please paste some text before running the check.")

    else:
        with st.spinner("🔎 Analyzing claims and checking evidence..."):

            try:
                result = run_pipeline(external_answer=text)

            except Exception:
                st.error(
                    "Something went wrong while analyzing the text. "
                    "Please check your pipeline and try again."
                )
                st.stop()

        st.divider()

        # -----------------------------
        # Score
        # -----------------------------
        score_data = result.get("score", {})
        score = score_data.get("overall_score")

        st.subheader("📊 Overall Result")

        if score is not None:

            # Convert 0-1 score to percentage if necessary
            if 0 <= score <= 1:
                percentage = score * 100
            else:
                percentage = score

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Hallucination Score",
                    f"{percentage:.1f}%",
                )

            with col2:
                claims = result.get("claims", [])
                st.metric(
                    "Claims Analyzed",
                    len(claims) if isinstance(claims, list) else "—",
                )

            with col3:
                st.metric(
                    "Risk Level",
                    (
                        "High"
                        if percentage >= 70
                        else "Medium"
                        if percentage >= 40
                        else "Low"
                    ),
                )

            st.progress(
                min(max(percentage / 100, 0.0), 1.0)
            )

        # -----------------------------
        # Claims
        # -----------------------------
        claims = result.get("claims", [])

        if claims:

            st.divider()
            st.subheader("🔬 Claim Analysis")

            # Summary counters
            supported = 0
            hallucinated = 0
            uncertain = 0

            for claim in claims:

                verdict = str(
                    claim.get("verdict", "")
                ).lower()

                if "support" in verdict or "true" in verdict:
                    supported += 1

                elif (
                    "halluc" in verdict
                    or "false" in verdict
                ):
                    hallucinated += 1

                else:
                    uncertain += 1

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("✅ Supported", supported)

            with c2:
                st.metric("❌ Potentially Hallucinated", hallucinated)

            with c3:
                st.metric("⚠️ Uncertain", uncertain)

            st.write("")

            # -----------------------------
            # Individual claims
            # -----------------------------
            for i, claim in enumerate(claims, start=1):

                claim_text = claim.get(
                    "claim",
                    claim.get("text", "Unknown claim"),
                )

                verdict = claim.get(
                    "verdict",
                    "Unknown",
                )

                confidence = claim.get(
                    "confidence"
                )

                evidence = claim.get(
                    "evidence"
                )

                verdict_lower = str(
                    verdict
                ).lower()

                if (
                    "support" in verdict_lower
                    or "true" in verdict_lower
                ):
                    icon = "✅"

                elif (
                    "halluc" in verdict_lower
                    or "false" in verdict_lower
                ):
                    icon = "❌"

                else:
                    icon = "⚠️"

                with st.expander(
                    f"{icon} Claim {i}: {claim_text}"
                ):

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Verdict**")
                        st.write(verdict)

                    with col2:
                        if confidence is not None:
                            st.markdown("**Confidence**")
                            st.write(confidence)

                    if evidence:
                        st.markdown("**Evidence**")
                        st.write(evidence)

        # -----------------------------
        # Raw result
        # -----------------------------
        with st.expander("🛠️ View raw pipeline result"):
            st.json(result)
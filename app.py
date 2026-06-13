import streamlit as st
from transformers import pipeline
import plotly.graph_objects as go

st.set_page_config(page_title="Sentiment Analyser", page_icon="🧠", layout="centered")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english")

analyser = load_model()

st.title("🧠 Sentiment Analyser")
st.markdown("Built with BERT + Streamlit | AI Portfolio Project")
st.markdown("---")

st.subheader("Analyse a single sentence")
user_text = st.text_area("Enter your text here:",
    placeholder="Type anything — a review, tweet, sentence...",
    height=100)

if st.button("Analyse Sentiment", type="primary"):
    if user_text.strip():
        with st.spinner("Analysing..."):
            result = analyser(user_text)[0]
            label = result["label"]
            score = round(result["score"] * 100, 1)

        col1, col2 = st.columns(2)
        with col1:
            colour = "green" if label == "POSITIVE" else "red"
            st.markdown(f"### Result: :{colour}[{label}]")
        with col2:
            st.metric("Confidence", f"{score}%")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score if label == "POSITIVE" else 100 - score,
            title={"text": "Positivity Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#22c55e" if label == "POSITIVE" else "#ef4444"},
                "steps": [
                    {"range": [0, 40], "color": "#fee2e2"},
                    {"range": [40, 60], "color": "#fef9c3"},
                    {"range": [60, 100], "color": "#dcfce7"},
                ],
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please enter some text first.")

st.markdown("---")
st.subheader("Analyse multiple texts at once")
bulk_text = st.text_area("Enter one sentence per line:",
    placeholder="Line 1 text here\nLine 2 text here\nLine 3 text here",
    height=150)

if st.button("Analyse All"):
    lines = [l.strip() for l in bulk_text.split("\n") if l.strip()]
    if lines:
        with st.spinner(f"Analysing {len(lines)} texts..."):
            results = analyser(lines)

        st.markdown("#### Results:")
        pos = sum(1 for r in results if r["label"] == "POSITIVE")
        neg = len(results) - pos

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(results))
        col2.metric("Positive", pos, delta=f"{round(pos/len(results)*100)}%")
        col3.metric("Negative", neg,
                    delta=f"-{round(neg/len(results)*100)}%",
                    delta_color="inverse")

        for line, result in zip(lines, results):
            label = result["label"]
            score = round(result["score"] * 100, 1)
            icon = "+" if label == "POSITIVE" else "-"
            colour = "green" if label == "POSITIVE" else "red"
            st.markdown(f"*[{icon}]* :{colour}[{label}] ({score}%) — {line}")
    else:
        st.warning("Please enter at least one line of text.")

st.markdown("---")
st.caption("Model: DistilBERT fine-tuned on SST-2 | Built for AI Portfolio")

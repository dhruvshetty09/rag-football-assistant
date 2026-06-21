"""
streamlit_app.py — The deployed front end.

Run locally:
  streamlit run app/streamlit_app.py

Deploy for free (Week 4):
  1. Push your repo to GitHub (without data/raw/ and .env)
  2. Go to share.streamlit.io and connect your repo
  3. Add your OPENAI_API_KEY in the Streamlit secrets panel

The retrieved chunks panel below is important — it's not just a debug tool.
Showing the user *what was retrieved and why* is a real UX feature, and
being able to point at it in an interview shows you think about transparency.
"""

import time
import streamlit as st
from agent.agent import run

st.set_page_config(page_title="RAG Research Assistant", layout="centered")

st.title("Research Assistant")
st.caption("Answers grounded in the document corpus, with citations.")

question = st.text_input("Ask a question:", placeholder="e.g. What is a denoising autoencoder?")

if st.button("Ask", type="primary") and question.strip():

    with st.spinner("Retrieving and thinking..."):
        t0 = time.time()
        result = run(question)
        latency = time.time() - t0

    # Main answer
    st.markdown("### Answer")
    st.write(result["answer"])

    # Confidence + latency
    col1, col2 = st.columns(2)
    col1.metric("Confidence", result.get("confidence", "—").capitalize())
    col2.metric("Latency", f"{latency:.1f}s")

    # Citations
    if result.get("citations"):
        st.markdown("**Sources cited:**")
        for cid in result["citations"]:
            st.markdown(f"- `{cid}`")

    if result.get("parse_error"):
        st.warning("Note: structured output parsing failed — answer may be less reliable.")

    # Retrieved chunks (expandable)
    with st.expander("Show retrieved chunks"):
        chunks = result.get("retrieved_chunks", [])
        if not chunks:
            st.write("No chunks retrieved.")
        for chunk in chunks:
            st.markdown(f"**{chunk['chunk_id']}** &nbsp; score: `{chunk['score']:.3f}`")
            st.text(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))
            st.divider()

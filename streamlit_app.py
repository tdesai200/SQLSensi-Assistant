import streamlit as st
import os
from source_scripts.rag_haiku_engine import (
    classify_sql,
    retrieve_rag_chunks,
    explain_with_haiku,
)

st.set_page_config(
    page_title="SQL Intelligence Assistant",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 SQL Intelligence Assistant")
st.markdown("""
This demo showcases a two-part system for SQL query analysis:

### 1 — SQL Classification  
Classifies queries using a trained **TF-IDF + SVM supervised model**.

### 2 — RAG-Powered SQL Explanation  
Retrieves SQL best-practice context using **embeddings + ChromaDB**,  
then generates a grounded explanation using **Claude 3 Haiku**.

Paste any SQL query below to begin.
""")

sql_input = st.text_area(
    "✏️ Enter your SQL query here:",
    height=200,
    placeholder="SELECT * FROM orders WHERE order_date > '2024-01-01';"
)

if st.button("🚀 Analyze SQL", type="primary"):
    if not sql_input.strip():
        st.error("Please enter a valid SQL query.")
    else:
        # -----------------------
        # 1 (Classifier)
        # -----------------------
        label = classify_sql(sql_input)
        st.subheader("🔍 1. Classification")
        st.write(f"**Predicted Label:** `{label}`")

        # -----------------------
        # 2 (RAG)
        # -----------------------
        st.subheader("📚 2. Retrieved Knowledge (RAG)")
        chunks = retrieve_rag_chunks(sql_input, label)

        if not chunks:
            st.warning("No relevant knowledge base chunks found.")
        else:
            for i, chunk in enumerate(chunks, 1):
                st.markdown(f"**Chunk {i}:**")
                st.info(chunk)

        # -----------------------
        # 3 (LLM Explanation)
        # -----------------------
        st.subheader("🤖 3. Claude 3 Haiku Explanation")
        explanation = explain_with_haiku(sql_input, label, chunks)
        st.success(explanation)

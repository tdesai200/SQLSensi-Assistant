"""
rag_haiku_engine.py
---------------------------------
Core engine for the SQL Intelligence Assistant.

Provides:
  • 1: classify_sql(sql)         -> TF-IDF + SVM label
  • 2: retrieve_rag_chunks(...)  -> embedding-based retrieval (Chroma)
  • 3: explain_with_haiku(...)   -> grounded explanation via Claude 3 Haiku

Environment:
  Requires ANTHROPIC_API_KEY to be set.

Usage:
  from source_scripts.rag_haiku_engine import classify_sql, retrieve_rag_chunks, explain_with_haiku
"""

import os
from typing import List

from joblib import load
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
import chromadb
from chromadb.config import Settings


# ---------------------------
# Path setup
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf.joblib")
SVM_PATH = os.path.join(MODEL_DIR, "svm.joblib")

# Lazy singletons
_VECTOR = None
_SVM = None
_EMBED = None
_CHROMA = None
_ANTHROPIC = None

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-3-haiku-20240307"


# ---------------------------
# Internal lazy loaders
# ---------------------------
def _load_vectorizer():
    global _VECTOR
    if _VECTOR is None:
        if not os.path.exists(TFIDF_PATH):
            raise FileNotFoundError("TF-IDF vectorizer not found. Run train_classifier.py first.")
        _VECTOR = load(TFIDF_PATH)
    return _VECTOR


def _load_svm():
    global _SVM
    if _SVM is None:
        if not os.path.exists(SVM_PATH):
            raise FileNotFoundError("SVM model not found. Run train_classifier.py first.")
        _SVM = load(SVM_PATH)
    return _SVM


def _load_embedder():
    global _EMBED
    if _EMBED is None:
        _EMBED = SentenceTransformer(EMBED_MODEL_NAME)
    return _EMBED


def _load_chroma():
    global _CHROMA
    if _CHROMA is None:
        if not os.path.isdir(CHROMA_DIR):
            raise FileNotFoundError("Chroma DB missing. Run build_kb_chroma.py first.")
        client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        _CHROMA = client.get_collection("sql_kb")
    return _CHROMA


def _load_anthropic():
    global _ANTHROPIC
    if _ANTHROPIC is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Use: $env:ANTHROPIC_API_KEY='sk-ant-...'"
            )
        _ANTHROPIC = Anthropic(api_key=api_key)
    return _ANTHROPIC


# ---------------------------
# PUBLIC API (used by CLI)
# ---------------------------
def classify_sql(sql_text: str) -> str:
    """1: Classify SQL query using TF-IDF + SVM."""
    if not sql_text.strip():
        raise ValueError("SQL text is empty.")

    vectorizer = _load_vectorizer()
    svm = _load_svm()
    X = vectorizer.transform([sql_text])
    return svm.predict(X)[0]


def retrieve_rag_chunks(sql_text: str, label: str, k: int = 3) -> List[str]:
    """Retrieve top-k relevant SQL knowledge base chunks using embeddings + Chroma."""
    embedder = _load_embedder()
    chroma = _load_chroma()

    query = (
        f"SQL best practices for {label} queries. "
        f"Consider the following SQL:\n{sql_text}"
    )

    embedding = embedder.encode([query]).tolist()
    result = chroma.query(query_embeddings=embedding, n_results=k)

    docs = result.get("documents", [[]])
    return docs[0] if docs else []


def explain_with_haiku(sql_text: str, label: str, context_chunks: List[str]) -> str:
    """2: Generate a grounded SQL explanation using Claude 3 Haiku."""

    context_text = "\n\n".join(context_chunks) if context_chunks else "(No context retrieved.)"

    # SAFE concatenated prompt
    prompt = (
        "You are a senior SQL performance engineer.\n"
        "Use ONLY the provided context. If something is unknown, say so.\n\n"

        f"SQL Query (Type: {label}):\n"
        "```\n"
        f"{sql_text}\n"
        "```\n\n"

        "Relevant SQL best-practice context:\n"
        "```\n"
        f"{context_text}\n"
        "```\n\n"

        "Explain clearly:\n"
        f"1) What this query is doing and why it was classified as {label}.\n"
        "2) Potential performance or style issues.\n"
        "3) Specific, actionable recommendations to improve it.\n"
    )

    client = _load_anthropic()
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=400,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        return resp.content[0].text
    except Exception:
        return str(resp)

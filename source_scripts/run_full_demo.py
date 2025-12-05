"""
run_full_demo.py
----------------
CLI demonstration for the SQL Intelligence Assistant.

Runs both:
  1. SQL classification (TF-IDF + SVM)
  2. RAG retrieval + Claude 3 Haiku explanation
"""

from rag_haiku_engine import (
    classify_sql,
    retrieve_rag_chunks,
    explain_with_haiku,
)

EXAMPLE_SQL = """
SELECT *
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_date >= '2024-01-01'
ORDER BY o.order_date DESC;
"""

def run_demo(sql):
    print("\n==============================")
    print("  SQL INTELLIGENCE ASSISTANT")
    print("==============================\n")

    print("➡ INPUT SQL QUERY:")
    print(sql)

    # -----------------------------
    # 1: CLASSIFICATION
    # -----------------------------
    label = classify_sql(sql)
    print("\n [1] Classification Result:")
    print(f"Predicted label: {label}")

    # -----------------------------
    # 2 (part A): RAG RETRIEVAL
    # -----------------------------
    print("\n [2] Retrieved Knowledge Chunks:")
    chunks = retrieve_rag_chunks(sql, label)
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)

    # -----------------------------
    # 3 (part B): CLAUDE EXPLANATION
    # -----------------------------
    print("\n [3] Claude 3 Haiku Explanation:")
    explanation = explain_with_haiku(sql, label, chunks)
    print("\n" + explanation + "\n")


if __name__ == "__main__":
    run_demo(EXAMPLE_SQL)

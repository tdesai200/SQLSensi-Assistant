import os
import glob
import uuid
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Base paths
BASE = os.path.dirname(os.path.dirname(__file__))
KB_DIR = os.path.join(BASE, "knowledge_base")
CHROMA_DIR = os.path.join(BASE, "chroma_db")

os.makedirs(CHROMA_DIR, exist_ok=True)

def read_documents():
    docs = []
    for path in glob.glob(os.path.join(KB_DIR, "*.md")):
        with open(path, "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks

def main():
    print("Loading knowledge base files...")
    docs = read_documents()

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))

    print(f"Total chunks: {len(all_chunks)}")

    print("Loading embedding model...")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Embedding chunks...")
    embeddings = embed_model.encode(all_chunks, show_progress_bar=True).tolist()

    print("Initializing Chroma DB...")
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_or_create_collection("sql_kb")

    ids = [str(uuid.uuid4()) for _ in all_chunks]

    print("Storing chunks in Chroma...")
    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=ids
    )

    print("Knowledge base successfully stored in chroma_db/")

if __name__ == "__main__":
    main()

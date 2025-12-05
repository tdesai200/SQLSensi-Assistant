# SQLSensi Assistant — INSTRUCTIONS

This document explains how to set up and run the demo CLI `run_full_demo.py` on Windows (PowerShell). The demo performs:

- a TF-IDF + SVM classification of a SQL query
- retrieval of knowledge chunks from the local Chroma DB
- a grounded explanation using Claude 3 Haiku (Anthropic)

Paths referenced in these instructions assume you run commands from the repository root (the folder that contains `source_scripts/`, `models/`, and `chroma_db/`).

**Prerequisites**

- Python 3.10 or 3.11 recommended.
- Git (optional).
- Internet access for model downloads and Anthropic API calls.
- An Anthropic API key if you want to call Claude: set `ANTHROPIC_API_KEY`.

**Install and prepare (PowerShell)**

1. Create and activate a virtual environment

```powershell
python -m venv .venv
# Activate for this PowerShell session:
.\.venv\Scripts\Activate.ps1
```

2. Upgrade `pip` (optional but recommended)

```powershell
python -m pip install --upgrade pip
```

3. Install dependencies

This project includes a `docs/requirements.txt` listing tested versions. Install it like:

```powershell
pip install -r docs/requirements.txt
```

Notes:
- Installation may download large packages (PyTorch, transformers, sentence-transformers). Expect some time and bandwidth.
- If you prefer a smaller footprint for local tests, install only the packages you need (for example `joblib`, `sentence-transformers`, and `chromadb`) — but full functionality requires the listed packages.

**Set Anthropic API Key (temporary for current PowerShell session)**

```powershell
$env:ANTHROPIC_API_KEY = 'sk-ant-...'
```

For a persistent environment variable on Windows, use `setx`, but note `setx` affects new shells only.

**Prepare models and knowledge base**

Before running the demo, ensure the TF-IDF + SVM models and Chroma DB exist. If the `models/` directory contains `tfidf.joblib` and `svm.joblib` and there is a `chroma_db/` directory with a Chroma collection, you can skip the building steps.

1. Train the classifier (if `models/tfidf.joblib` or `models/svm.joblib` are missing)

```powershell
python source_scripts\train_classifier.py
```

This reads `data/merged_sql_10000.csv` and writes `models/tfidf.joblib` and `models/svm.joblib`.

2. Build the Chroma knowledge base (if `chroma_db/` is missing or empty)

```powershell
python source_scripts\build_kb_chroma.py
```

This will:
- read `.md` files from `knowledge_base/`, chunk and embed them,
- create a persistent Chroma DB at `chroma_db/` and a collection named `sql_kb`.

**Run the demo CLI**

From the project root run:

```powershell
python source_scripts\run_full_demo.py
```

What this does:
- prints the input SQL (use the example SQL embedded in `run_full_demo.py`),
- runs classification (`classify_sql`),
- retrieves related KB chunks from Chroma (`retrieve_rag_chunks`),
- asks Anthropic Claude 3 Haiku for a grounded explanation (`explain_with_haiku`).

If you want to run the same demo repeatedly while modifying the sample SQL, edit `EXAMPLE_SQL` inside `source_scripts/run_full_demo.py` or create a simple wrapper script to call `run_demo(...)`.

**Running the Streamlit app (optional)**

The repo includes a `streamlit_app.py`. To run the UI:

```powershell
streamlit run streamlit_app.py
```

**Common issues & troubleshooting**

- FileNotFoundError: "TF-IDF vectorizer not found. Run train_classifier.py first." — run `python source_scripts\train_classifier.py`.
- FileNotFoundError: "Chroma DB missing. Run build_kb_chroma.py first." — run `python source_scripts\build_kb_chroma.py`.
- EnvironmentError: "ANTHROPIC_API_KEY not set." — set the env var in PowerShell with `$env:ANTHROPIC_API_KEY = 'sk-...'`.
- Long downloads or Cuda issues: `sentence-transformers` and `torch` may attempt to use GPU; ensure appropriate torch install for your platform or use CPU-only.

**Security & cost notes**

- The demo calls the Anthropic Claude API. Do not commit your API keys to source control. Be aware of API usage costs and rate limits.
- The prompt sent to Claude is constructed using only the provided context; however, always review outputs before acting on advice in production systems.

**Developer tips & advanced usage**

- To regenerate the classifier with different TF-IDF settings, modify `source_scripts/train_classifier.py` and re-run it.
- To add or update knowledge base content, edit or add `.md` files in `knowledge_base/` and re-run `build_kb_chroma.py`.
- If you want to run the demo code from another script, you can import and call `run_demo(sql)` by executing the module file (`python source_scripts\run_full_demo.py`) or by importing via importlib from its path.

**Core engine (`rag_haiku_engine.py`)**

- Location: `source_scripts/rag_haiku_engine.py`.
- Public functions used by the demo:
	- `classify_sql(sql_text)` — uses TF-IDF + SVM to predict a label.
	- `retrieve_rag_chunks(sql_text, label, k=3)` — embeds a query and retrieves top-k documents from the local Chroma DB.
	- `explain_with_haiku(sql_text, label, context_chunks)` — calls Anthropic Claude 3 Haiku to generate a grounded explanation using only retrieved context.
- Important implementation details and requirements:
	- The module expects model files at `models/tfidf.joblib` and `models/svm.joblib`. Run `source_scripts/train_classifier.py` to create them if missing.
	- The Chroma DB is expected at `chroma_db/` with a collection named `sql_kb`. Run `source_scripts/build_kb_chroma.py` to build it from the `knowledge_base/` markdown files.
	- It uses a local sentence-transformers embedding model (`sentence-transformers/all-MiniLM-L6-v2`) which will be downloaded on first use.
	- It requires the environment variable `ANTHROPIC_API_KEY` to be set for calls to Claude; otherwise `explain_with_haiku` raises an error.
	- The Anthropic model used by default is `claude-3-haiku-20240307` and requests are sent via the `anthropic` Python client.

**Files of interest**

- `source_scripts/run_full_demo.py` — main CLI demo runner
- `source_scripts/rag_haiku_engine.py` — core functions (classification, retrieval, Anthropic call)
- `source_scripts/train_classifier.py` — trains and saves `models/tfidf.joblib` and `models/svm.joblib`
- `source_scripts/build_kb_chroma.py` — builds `chroma_db/` from `knowledge_base/*.md`
- `docs/requirements.txt` — pinned dependencies used for this project
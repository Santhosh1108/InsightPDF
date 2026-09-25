# InsightPDF — Advanced Hybrid RAG

InsightPDF is a PDF question-answering application that now uses a **multi-stage RAG pipeline** instead of sending the first vector-search results directly to the LLM.

The retrieval layer is designed to reduce noisy context before generation:

**PDF → clean/chunk → dense retrieval + BM25 → Reciprocal Rank Fusion → cross-encoder reranking → duplicate filtering → MMR diversification → grounded LLM response**

## What changed

### 1. Hybrid retrieval
Semantic FAISS search is combined with BM25 lexical search. This makes the system better at both conceptual questions and exact terms such as product names, IDs, dates, section titles and technical phrases.

### 2. Reciprocal Rank Fusion
Dense and sparse rankings are fused without assuming their raw score scales are comparable.

### 3. Cross-encoder reranking
The top hybrid candidates are scored jointly against the query using:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

This is intentionally applied only to a bounded candidate pool rather than the whole document, keeping latency manageable.

### 4. Duplicate suppression
Highly similar chunks are removed before generation so the context budget is not wasted repeating the same passage.

### 5. MMR diversification
Maximal Marginal Relevance selects the final context to balance relevance with coverage across different parts of the document.

### 6. Grounded generation
Every retrieved passage is passed to the LLM with a source/document and page label. The generation prompt explicitly treats document text as evidence and rejects instructions embedded inside retrieved content.

### 7. Correct cache invalidation
The index fingerprint now includes retrieval settings, so changing chunk size, overlap or retrieval configuration cannot accidentally reuse a stale index.

## Architecture

```
                         PDF(s)
                           │
                           ▼
                 Text extraction + cleaning
                           │
                           ▼
                 Semantic-aware chunking
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        Sentence embeddings           BM25
              │                         │
              ▼                         ▼
          FAISS top-N                BM25 top-N
              └────────────┬────────────┘
                           ▼
                   Reciprocal Rank
                      Fusion (RRF)
                           │
                           ▼
                  Candidate filtering
                           │
                           ▼
                 Cross-encoder reranker
                           │
                           ▼
              Duplicate suppression
                           │
                           ▼
                  MMR diversification
                           │
                           ▼
                 Final evidence set
                           │
                           ▼
                  Groq Llama model
                           │
                           ▼
                Grounded answer + sources
```

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF parsing | PyMuPDF / pypdf |
| Chunking | Custom recursive sentence-aware splitter |
| Dense retrieval | Sentence Transformers + FAISS |
| Sparse retrieval | BM25 |
| Rank fusion | Reciprocal Rank Fusion |
| Reranking | Sentence Transformers CrossEncoder |
| Diversity | MMR |
| Generation | Groq + Llama |
| Language | Python |

## Run locally

```bash
git clone https://github.com/Santhosh1108/InsightPDF.git
cd InsightPDF
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Create `.env`:

```env
GROQ_API_KEY=your_api_key_here
```

Run:

```bash
streamlit run app.py
```

## Tuning

- **Final chunks:** how much evidence reaches the LLM.
- **Candidate pool:** how many hybrid results enter the reranking stage.
- **Cross-encoder reranking:** improves precision at the cost of additional local inference.
- **Diversity / relevance:** controls MMR. Higher values favor the most relevant passages; lower values increase coverage.
- **Minimum reranker score:** raises the relevance gate when strict retrieval is preferred.

For a latency-sensitive deployment, disable the cross-encoder. For quality-focused evaluation, keep it enabled with a candidate pool around 20–40.

## Project structure

```
InsightPDF/
├── app.py
├── requirements.txt
├── README.md
└── utils/
    ├── advanced_retriever.py
    ├── llm.py
    ├── pdf_loader.py
    ├── prompts.py
    ├── text_splitter.py
    └── vector_store.py
```

`utils/vector_store.py` remains as a compatibility entry point while the implementation lives in `utils/advanced_retriever.py`.

## Why this is more than vector search

The important change is that the LLM no longer receives the nearest chunks blindly. Retrieval is treated as a ranking pipeline:

**retrieve broadly → combine independent signals → score relevance → remove redundancy → optimize context diversity → generate**

This keeps the expensive generation step focused on a smaller, higher-quality evidence set.

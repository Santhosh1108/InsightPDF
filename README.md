# AI Support Agent

A Retrieval-Augmented Generation (RAG) application that lets users chat with one or more PDF documents using semantic search.

Instead of sending an entire document to an LLM, the application retrieves only the most relevant sections using vector embeddings and FAISS, then uses a Groq-hosted Llama model to generate grounded responses with source citations.

---

## Demo

<img width="1905" height="926" alt="image" src="https://github.com/user-attachments/assets/facc2520-794d-4d44-9bd7-3ef1be79df17" />


```
assets/
├── home.png
├── chat.png
└── sources.png
```

---

## Features

- Upload one or multiple PDF documents
- Semantic search using Sentence Transformers
- FAISS vector database for fast retrieval
- Groq + Llama 3 for answer generation
- Streaming responses
- Page-level source citations
- Recursive sentence-aware chunking
- Configurable retrieval settings
- Cached vector index for faster conversations
- Download chat history
- Reset index without restarting the application

---

## Architecture

```

                PDF Documents
                      │
                      ▼
             PDF Text Extraction
                      │
                      ▼
          Recursive Text Chunking
                      │
                      ▼
      Sentence Transformer Embeddings
                      │
                      ▼
                 FAISS Index
                      │
              User Question
                      │
                      ▼
            Semantic Retrieval
                      │
                      ▼
             Groq Llama 3 Model
                      │
                      ▼
      Streaming Response + Citations

```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| LLM | Groq (Llama 3.3) |
| PDF Processing | PyMuPDF / pypdf |
| Language | Python |

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/ai-support-agent.git

cd ai-support-agent
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

Windows

```bash
venv\Scripts\activate
```

macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
GROQ_API_KEY=your_api_key_here
```

Run the application

```bash
streamlit run app.py
```

---

## Project Structure

```

AI Support Agent/
│
├── app.py
├── requirements.txt
├── README.md
│
├── utils/
│   ├── llm.py
│   ├── pdf_loader.py
│   ├── text_splitter.py
│   └── vector_store.py
│
├── assets/
├── data/
└── vector_db/

```

---

## Design Decisions

A few implementation choices were made to improve both performance and usability.

### Cached Indexing

The application fingerprints uploaded documents and rebuilds the FAISS index only when the uploaded files change. This avoids recomputing embeddings on every Streamlit rerun.

### Recursive Chunking

Instead of splitting documents at fixed character counts, text is recursively divided by paragraphs, lines, sentences and finally words. This helps preserve semantic context.

### Source Attribution

Each retrieved chunk retains its page number and source document, allowing generated answers to reference where information originated.

### Streaming Responses

Responses are streamed token-by-token from the language model, providing faster perceived response times.

---

## Future Improvements

- OCR support for scanned PDFs
- Hybrid keyword + vector search
- Persistent vector database
- User authentication
- Conversation memory across sessions
- Cloud deployment

---

## Acknowledgements

This project uses:

- Streamlit
- FAISS
- Sentence Transformers
- Groq
- Llama 3
- PyMuPDF

---

## License

This project is licensed under the MIT License.

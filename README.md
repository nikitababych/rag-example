# RAG Chatbot

A local RAG (Retrieval-Augmented Generation) chatbot that answers questions based on a PDF and audio lecture. Built with LangChain.

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com) installed and running with `llama3` model
- [ffmpeg](https://ffmpeg.org/download.html) installed and on PATH (required for audio transcription)

## Setup

1. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

2. **Pull the LLM model via Ollama**
   ```
   ollama pull llama3
   ```

3. **Add your data files** to the `data/` folder.

## Usage

### First run — ingest + chat
Transcribes audio, processes PDF, builds the vector DB, then starts the chatbot:
```
python rag_chatbot.py
```
> ⚠️ First run takes several minutes — Whisper needs to transcribe the audio file.

### Chat only (after ingestion)
Skips ingestion and goes straight to the chatbot:
```
python rag_chatbot.py chat
```

### Inspect the vector DB
See what's stored in ChromaDB:
```
python inspect_db.py
```

## How it works

1. **Load** — extracts text from PDF (per page) and transcribes audio via Whisper
2. **Chunk** — PDF: merges short slides with neighbors to preserve context; Audio: semantic chunking (LangChain `SemanticChunker`) with a max-size second pass to keep chunks focused
3. **Embed** — converts chunks to vectors using `all-MiniLM-L6-v2` (local, free)
4. **Store** — saves embeddings to ChromaDB via LangChain Chroma integration
5. **Retrieve** — multi-query retriever generates alternative phrasings of the question via the LLM, then retrieves top-8 results per variant and deduplicates
6. **Generate** — LangChain LCEL chain passes question + context to Ollama (`ChatOllama`) for the final answer

## Key dependencies

- **LangChain** — orchestration, retrieval, prompt templates, LCEL chains
- **LangChain Experimental** — `SemanticChunker` for meaning-based text splitting
- **LangChain Chroma** — ChromaDB vector store integration
- **LangChain Ollama** — local LLM via `ChatOllama`
- **LangChain HuggingFace** — `HuggingFaceEmbeddings` for local sentence-transformers
- **Whisper** — audio transcription
- **PyMuPDF** — PDF text extraction

## Notes

- Everything runs **100% locally** — no API keys, no cost
- ChromaDB stores data in the `chroma_db/` folder (auto-created)
- To rebuild the vector DB, just run the full ingest again — it clears and rebuilds automatically
- Multi-query retrieval adds a small latency overhead (one extra LLM call per question) but significantly improves recall

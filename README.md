# RAG Chatbot

A local RAG (Retrieval-Augmented Generation) chatbot that answers questions based on a PDF and audio lecture.

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

1. **Load** — extracts text from PDF and transcribes audio via Whisper
2. **Chunk** — splits text into ~120-word overlapping chunks
3. **Embed** — converts chunks to vectors using `all-MiniLM-L6-v2` (local, free)
4. **Store** — saves embeddings to ChromaDB (local, no server needed)
5. **Retrieve** — finds the 5 most relevant chunks for each question
6. **Generate** — passes question + context to Ollama (local LLM) for the final answer

## Notes

- Everything runs **100% locally** — no API keys, no cost
- ChromaDB stores data in the `chroma_db/` folder (auto-created)
- To rebuild the vector DB, just run the full ingest again — it clears and rebuilds automatically

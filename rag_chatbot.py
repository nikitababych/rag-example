import glob
import fitz  # PyMuPDF
import whisper
import chromadb
from sentence_transformers import SentenceTransformer
import requests

# --- Config ---
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

PDF_EXTENSIONS = {"pdf"}
AUDIO_EXTENSIONS = {"mp3", "mp4", "m4a", "wav", "ogg", "flac"}


def scan_data_dir(directory: str) -> tuple[list[str], list[str]]:
    pdfs, audios = [], []
    for path in glob.glob(f"{directory}/**/*", recursive=True):
        ext = path.rsplit(".", 1)[-1].lower()
        if ext in PDF_EXTENSIONS:
            pdfs.append(path)
        elif ext in AUDIO_EXTENSIONS:
            audios.append(path)
    return pdfs, audios
COLLECTION_NAME = "rag_knowledge_base"
CHUNK_SIZE = 120
CHUNK_OVERLAP = 20
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3"

# --- 1. Load & Process Data ---

def extract_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def transcribe_audio(path: str) -> str:
    model = whisper.load_model("base")
    result = model.transcribe(path)
    return result["text"]


# --- 2. Chunk Text ---

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i : i + size])
        if chunk:
            chunks.append(chunk)
    return chunks


# --- 3. Embed & Store ---

def build_vector_store(chunks: list[str], sources: list[str]) -> chromadb.Collection:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # delete existing collection to rebuild
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(COLLECTION_NAME)
    embeddings = embedder.encode(chunks).tolist()

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"source": s} for s in sources],
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB.")
    return collection


# --- 4. Retrieve & Generate ---

def retrieve(query: str, collection: chromadb.Collection, n_results: int = 5) -> list[str]:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    return results["documents"][0]


def generate_answer(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": "Answer the question based only on the provided context. If the context doesn't contain the answer, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    })
    return response.json()["message"]["content"]


# --- Main ---

def ingest():
    """Load sources, chunk, embed, and store."""
    pdfs, audios = scan_data_dir(DATA_DIR)
    print(f"Found {len(pdfs)} PDF(s) and {len(audios)} audio file(s) in '{DATA_DIR}/'")

    all_chunks, sources = [], []

    for path in pdfs:
        print(f"Extracting PDF: {path}")
        chunks = chunk_text(extract_pdf_text(path))
        all_chunks += chunks
        sources += [f"pdf:{path}"] * len(chunks)

    for path in audios:
        print(f"Transcribing audio: {path} (this may take a while)...")
        chunks = chunk_text(transcribe_audio(path))
        all_chunks += chunks
        sources += [f"audio:{path}"] * len(chunks)

    print(f"Created {len(all_chunks)} total chunks.")
    build_vector_store(all_chunks, sources)


def chat():
    """Interactive Q&A loop over the ingested knowledge base."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    print("\nRAG Chatbot ready. Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        context = retrieve(question, collection)
        answer = generate_answer(question, context)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        chat()
    else:
        ingest()
        chat()

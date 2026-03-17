import glob
import fitz  # PyMuPDF
import whisper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Config ---
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_knowledge_base"
OLLAMA_MODEL = "llama3"
MAX_CHUNK_CHARS = 1500

PDF_EXTENSIONS = {"pdf"}
AUDIO_EXTENSIONS = {"mp3", "mp4", "m4a", "wav", "ogg", "flac"}

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# --- 0. Scan data dir ---

def scan_data_dir(directory: str) -> tuple[list[str], list[str]]:
    pdfs, audios = [], []
    for path in glob.glob(f"{directory}/**/*", recursive=True):
        ext = path.rsplit(".", 1)[-1].lower()
        if ext in PDF_EXTENSIONS:
            pdfs.append(path)
        elif ext in AUDIO_EXTENSIONS:
            audios.append(path)
    return pdfs, audios


# --- 1. Load & Process Data ---

def extract_pdf_pages(path: str, min_chars: int = 200) -> list[Document]:
    """Extract text per page, merging short pages (titles/dividers) into the next page."""
    doc = fitz.open(path)
    raw = [(page.get_text().strip(), i + 1) for i, page in enumerate(doc)]

    pages = []
    carry = ""
    carry_page = None
    for text, page_num in raw:
        if not text:
            continue
        if carry:
            text = carry + "\n" + text
            page_num = carry_page
            carry = ""
        if len(text) < min_chars:
            carry = text
            carry_page = page_num
        else:
            pages.append(Document(page_content=text, metadata={"source": f"pdf:{path}", "page": page_num}))
    if carry:
        pages.append(Document(page_content=carry, metadata={"source": f"pdf:{path}", "page": carry_page}))
    return pages


def transcribe_audio(path: str) -> str:
    model = whisper.load_model("base")
    result = model.transcribe(path)
    return result["text"]


# --- 2. Semantic Chunking & Vector Store ---

def build_vector_store(docs: list[Document]) -> Chroma:
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass
    print(f"Indexing {len(docs)} documents.")
    return Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR, collection_name=COLLECTION_NAME)


# --- 3. RAG Chain ---

def build_rag_chain(vector_store: Chroma):
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever

    base_retriever = vector_store.as_retriever(search_kwargs={"k": 8})
    llm = ChatOllama(model=OLLAMA_MODEL)
    retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using the provided context. Synthesize information from multiple context passages when needed. If the context truly contains no relevant information, say so."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    def format_docs(docs):
        return "\n\n---\n\n".join(d.page_content for d in docs)

    return {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm


# --- Main ---

def ingest():
    pdfs, audios = scan_data_dir(DATA_DIR)
    print(f"Found {len(pdfs)} PDF(s) and {len(audios)} audio file(s) in '{DATA_DIR}/'")

    all_docs = []

    for path in pdfs:
        print(f"Extracting PDF: {path}")
        all_docs.extend(extract_pdf_pages(path))

    for path in audios:
        print(f"Transcribing audio: {path} (this may take a while)...")
        text = transcribe_audio(path)
        chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_CHARS, chunk_overlap=200)
        chunks = chunker.create_documents([text])
        for chunk in chunks:
            if len(chunk.page_content) > MAX_CHUNK_CHARS:
                sub_chunks = splitter.split_documents([chunk])
                for sc in sub_chunks:
                    sc.metadata["source"] = f"audio:{path}"
                all_docs.extend(sub_chunks)
            else:
                chunk.metadata["source"] = f"audio:{path}"
                all_docs.append(chunk)

    print(f"Created {len(all_docs)} total documents.")
    return build_vector_store(all_docs)


def chat(vector_store: Chroma = None):
    if vector_store is None:
        vector_store = Chroma(persist_directory=CHROMA_DIR, collection_name=COLLECTION_NAME, embedding_function=embeddings)

    chain = build_rag_chain(vector_store)
    print("\nRAG Chatbot ready. Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        response = chain.invoke(question)
        print(f"\nAssistant: {response.content}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        chat()
    else:
        vs = ingest()
        chat(vs)

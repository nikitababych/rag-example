from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

store = Chroma(persist_directory="chroma_db", collection_name="rag_knowledge_base",
               embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

docs = store.get(include=["documents", "metadatas", "embeddings"], limit=15)
print(f"Total chunks: {len(docs['ids'])}\n")

for i in range(len(docs["ids"])):
    print(f"--- {docs['ids'][i]} ({docs['metadatas'][i].get('source', 'unknown')}) ---")
    print(f"Text: {docs['documents'][i][:200]}...")
    print(f"Embedding (first 10 dims): {docs['embeddings'][i][:10]}")
    print()

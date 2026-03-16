import chromadb

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("rag_knowledge_base")

print(f"Total chunks: {collection.count()}\n")

results = collection.get(include=["documents", "metadatas", "embeddings"], limit=15)

for i in range(len(results["ids"])):
    print(f"--- {results['ids'][i]} ({results['metadatas'][i]['source']}) ---")
    print(f"Text: {results['documents'][i][:200]}...")
    print(f"Embedding (first 10 dims): {results['embeddings'][i][:10]}")
    print()

import json
from tqdm import tqdm
from src.ingestion.parser import BEIRParser
from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore

parser = BEIRParser()

print("Parsing and chunking documents...")
chunks = parser.parse()
print(f"Total chunks: {len(chunks)}")

# save chunks for BM25
with open("data/chunks.json", "w") as f:
    json.dump(chunks, f)
print("Chunks saved.")

embedder = Embedder()
vectors = []
print("Embedding chunks...")
for i in tqdm(range(0, len(chunks), 8), desc="Embedding"):
    batch = chunks[i:i+8]
    texts = [c["chunk"] for c in batch]
    batch_vectors = list(embedder.embedding_model.embed(texts))
    vectors.extend(batch_vectors)

print("Upserting to Qdrant...")
store = VectorStore()
store.create_collection()

batch_size = 100
for i in tqdm(range(0, len(chunks), batch_size), desc="Upserting"):
    store.upsert(chunks[i:i+batch_size], vectors[i:i+batch_size])

print(f"Done. Ingested {len(chunks)} chunks.")
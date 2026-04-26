from src.ingestion.parser import BEIRParser
from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.fusion import RRF

query = "What is the best way to invest in stocks?"

# Step 1 - check dense retrieval
dense = DenseRetriever()
dense_results = dense.retrieve(query, top_k=5)
print("=== DENSE RESULTS ===")
for doc in dense_results:
    print(doc.payload['chunk'][:200])
    print("---")

# Step 2 - check sparse retrieval
parser = BEIRParser()
sparse = SparseRetriever(parser.parse())
sparse_results = sparse.retrieve(query, top_k=5)
print("=== SPARSE RESULTS ===")
for doc in sparse_results:
    print(doc['chunk'][:200])
    print("---")

# Step 3 - check fusion
rrf = RRF()
fused = rrf.calculateRPF(query, top_k=5)
print("=== FUSED RESULTS ===")
for doc in fused:
    print(doc['chunk'][:200])
    print("---")
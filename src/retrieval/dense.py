from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore

class DenseRetriever:

    def __init__(self):
        self.embedder=Embedder()
        self.vectorstore=VectorStore()
        self.top_k=20

    def retrieve(self,query:str,top_k:int)->list:
        vector=self.embedder.embed_query(query)
        return self.vectorstore.search(vector,top_k=self.top_k)
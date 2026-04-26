from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore

class DenseRetriever:

    def __init__(self):
        self.embedder=Embedder()
        self.vectorstore=VectorStore()

    def retrieve(self,query:str,top_k:int)->list:
        vector=self.embedder.embed_query(query)
        return self.vectorstore.search(vector,top_k=top_k)
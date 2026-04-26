from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever


class RPF:
    def __init__(self):
        self.dense_retriever=DenseRetriever()
        self.sparse_retriever=SparseRetriever()

    def calculateRPF(self,query:str,top_k:int=20):
        scores={}
        dense_docs=self.dense_retriever.retrieve(query,top_k=top_k)
        sparse_docs=self.sparse_retriever.retrieve(query,top_k=top_k)

        doc_map = {}  

        for i, doc in enumerate(dense_docs):
            chunk_id = doc.payload["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + 1/(60+i)
            doc_map[chunk_id] = doc.payload  

        for i, doc in enumerate(sparse_docs):
            chunk_id = doc["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + 1/(60+i)
            doc_map[chunk_id] = doc  

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_map[chunk_id] for chunk_id, _ in sorted_scores[:top_k]]


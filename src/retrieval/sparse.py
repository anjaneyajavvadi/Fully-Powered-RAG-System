from rank_bm25 import BM25Okapi
from datasets import load_from_disk

class SparseRetriever:
    def __init__(self,chunks:list[dict]):
        self.chunks = chunks
        split_docs = [c["chunk"].lower().split(" ") for c in chunks]
        self.bm25 = BM25Okapi(split_docs)


    def retrieve(self, query: str, top_k: int) -> list[dict]:
        q = query.lower().split(" ")
        top_texts = self.bm25.get_top_n(q, self.chunks, n=top_k)
        return top_texts

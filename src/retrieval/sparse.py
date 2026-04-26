from rank_bm25 import BM25Okapi
import json

class SparseRetriever:
    def __init__(self):
        with open("data/chunks.json", "r") as f:
            self.chunks = json.load(f)
        split_docs = [c["chunk"].lower().split(" ") for c in self.chunks]
        self.bm25 = BM25Okapi(split_docs)


    def retrieve(self, query: str, top_k: int) -> list[dict]:
        q = query.lower().split(" ")
        top_texts = self.bm25.get_top_n(q, self.chunks, n=top_k)
        return top_texts

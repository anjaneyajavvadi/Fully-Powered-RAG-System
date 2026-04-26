from rank_bm25 import BM25Okapi
from datasets import load_from_disk

class SparseRetrieve:
    def __init__(self):
        self.bm25=None
        self.data_path='../data/raw/fiqa_corpus'
        self.dataset=load_from_disk(self.data_path)

    def BM25store(self):
        all_docs=[]
        for doc in self.dataset['corpus']:
            all_docs.append(doc['text'])

        split_docs=[text.lower().split(" ") for text in all_docs]
        self.bm25=BM25Okapi(split_docs)

    def retrieve(self,query:str,top_k:int):
        q=query.lower().split(" ")
        return self.bm25.get_top_n(q,n=top_k)

from sentence_transformers import CrossEncoder
from utils.yaml_loader import load_config
from src.retrieval.fusion import RRF

class ReRanker:
    def __init__(self):
        self.config=load_config()
        self.rrf=RRF()
        self.model_name=self.config['crossencoder']['model']
        self.encoder=CrossEncoder(self.model_name)

    def rerank(self, query: str):
        docs = self.rrf.fused_retrieve(query=query, top_k=10)

        pairs = [(query, doc['chunk']) for doc in docs]
        scores = self.encoder.predict(pairs)

        top_k = []
        for i, doc in enumerate(docs):
            top_k.append({"score": scores[i], **doc})

        top_k.sort(key=lambda x: x["score"], reverse=True)
        return top_k[:5]

from sentence_transformers import SentenceTransformer
from src.utils.yaml_loader import load_config
from src.utils.logger import get_logger

class Embedder:
    def __init__(self):
        self.config=load_config()
        self.embedding_model=SentenceTransformer(model_name=self.config['embedding']['model'])
        self.logger = get_logger(__name__)
        self.logger.info("Embedder initialized")

    def embed_docs(self, chunks:list[dict]):
        self.logger.info("Embedder embedding docs")
        embedded_docs=[]
        for chunk in chunks:
            embedded_docs.append(list(self.embedding_model.embed(chunk['chunk']))[0])
        return embedded_docs

    def embed_query(self,text:str):
        self.logger.info("Embedding query")
        return list(self.embedding_model.embed([text]))[0]
    
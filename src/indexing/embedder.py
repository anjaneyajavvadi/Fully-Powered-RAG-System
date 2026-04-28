from sentence_transformers import SentenceTransformer
from src.utils.yaml_loader import load_config
from src.utils.logger import get_logger

class Embedder:
    def __init__(self):
        self.config=load_config()
        self.embedding_model=SentenceTransformer(self.config['embedding']['model'])
        self.logger = get_logger(__name__)
        self.logger.info("Embedder initialized")

    def embed_docs(self, chunks: list[dict]):
        self.logger.info("Embedding docs")
        texts = [chunk['chunk'] for chunk in chunks]
        vectors = self.embedding_model.encode(
            texts,
            batch_size=8,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return vectors.tolist()

    def embed_query(self, text: str):
        self.logger.info("Embedding query")
        return self.embedding_model.encode(
            text,
            normalize_embeddings=True
        ).tolist()
    
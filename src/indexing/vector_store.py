from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.utils.yaml_loader import load_config
from src.utils.logger import get_logger
import uuid

class VectorStore:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.logger.info("Embedder initialized")
        self.config=load_config()
        self.port=self.config['qdrant']['port']
        self.host=self.config['qdrant']['host']
        self.collection_name=self.config['qdrant']['collection_name']
        self.client=QdrantClient(url=f"http://{self.host}:{self.port}")
        self.logger = get_logger(__name__)
        self.logger.info("VectorStore initialized")

    def create_collection(self):
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.config['qdrant']['size'],
                distance=models.Distance.COSINE
            )
        )
        self.logger.info("Collection Created")

    def upsert(self,chunks:list[str],vectors:list):
        points=[]
        for i,(chunk,vector) in enumerate(zip(chunks,vectors)):
            points.append(
                models.PointStruct(
                    id=uuid.uuid4(),
                    vector=vector,
                    payload={
                        "title": chunk["title"],
                        "text": chunk["text"],
                        "chunk": chunk["chunk"]
                    }
                )
            )
        self.logger.info("starting upsert")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        self.logger.info("Completed upsert")
    def search(self,query_vector:list,top_k:int):
        self.logger.info("Seaching Query")
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        )

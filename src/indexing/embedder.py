from fastembed import TextEmbedding
from utils.yaml_loader import load_config

class Embedder:
    def __init__(self):
        self.config=load_config()
        self.embedding_model=TextEmbedding(model_name=self.config['embedding']['model'])

    def embed_docs(self, chunks:list[dict]):
        embedded_docs=[]
        for chunk in chunks:
            embedded_docs.append(list(self.embedding_model.embed(chunk['chunk']))[0])
        return embedded_docs

    def embed_query(self,text:str):
        return list(self.embedding_model.embed([text]))[0]
    
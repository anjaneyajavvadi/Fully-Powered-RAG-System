from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
import yaml
from utils.yaml_loader import load_config
class ContextAwareChunker:
    def __init__(self):
        self.yaml_config=load_config()
        self.embedding_model=HuggingFaceEmbeddings(model_name=self.yaml_config['chunking']['model_name'])
        self.semantic_chunker=SemanticChunker(embeddings=self.embedding_model,breakpoint_threshold_amount=self.yaml_config['chunking']['breakpoint_threshold'])
        self.text_splitter=RecursiveCharacterTextSplitter(chunk_size=self.yaml_config['chunking']['chunk_size'],chunk_overlap=self.yaml_config['chunking']['chunk_overlap'])

    def semantic_chunk(self,text:str)->List[str]:
        return self.semantic_chunker.split_text(text)

    def recursive_chunk(self,text:str)->List[str]:
        return self.text_splitter.split_text(text)

    def chunk(self,text:str,strategy:str='semantic')->List[str]:
        if strategy=='semantic':
            chunks=self.semantic_chunk(text)
            final_chunks=[]
            for chunk in chunks:
                if len(chunk)>self.yaml_config['chunking']['chunk_size']:
                    final_chunks.extend(self.recursive_chunk(chunk))
                else:
                    final_chunks.append(chunk)
            return final_chunks
        else:
            return self.recursive_chunk(text)
        
    
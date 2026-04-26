from datasets import load_from_disk
from src.ingestion.chunker import ContextAwareChunker

class BEIRParser:
    def __init__(self):
        self.data_path='../data/raw/fiqa_corpus'
        self.chunker=ContextAwareChunker()
        self.load()

    def load(self):
        self.dataset=load_from_disk(self.data_path)

    def parse(self):
        parsed_chunks=[]
        for doc in self.dataset['corpus']:
            id=doc['id']
            title=doc['title']
            text=doc['text']

            chunks=self.chunker.chunk(text)
            for i,chunk in enumerate(chunks):
                chunk_dict={"chunk_id": f"{id}_{j}",'id':id,'title':title,'text':text,'chunk':chunk}
                parsed_chunks.append(chunk_dict)

        return parsed_chunks

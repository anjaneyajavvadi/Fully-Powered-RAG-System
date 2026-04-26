from datasets import load_from_disk
from src.ingestion.chunker import ContextAwareChunker
from src.utils.logger import get_logger

class BEIRParser:
    def __init__(self):
        self.data_path='data/raw/fiqa_corpus'
        self.chunker=ContextAwareChunker()
        self.logger = get_logger(__name__)
        self.logger.info("Parser initialized")
        self.load()
        


    def load(self):
        self.dataset=load_from_disk(self.data_path)
        self.logger.info("Loading Data")
        self.dataset['corpus'] = self.dataset['corpus'].select(range(2000))

    def parse(self):
        parsed_chunks=[]
        self.logger.info("parsing Data")
        for doc in self.dataset['corpus']:
            id=doc['_id']
            title=doc['title']
            text=doc['text']

            chunks=self.chunker.chunk(text)
            for i,chunk in enumerate(chunks):
                chunk_dict={"chunk_id": f"{id}_{i}",'id':id,'title':title,'text':text,'chunk':chunk}
                parsed_chunks.append(chunk_dict)
        self.logger.info("parsing Completed")

        return parsed_chunks

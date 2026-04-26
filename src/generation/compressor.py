from llmlingua import PromptCompressor
from utils.yaml_loader import load_config

class Compressor:
    def __init__(self):
        self.config=load_config()
        self.compressor=PromptCompressor(self.config['compression']['model'])
        
    def compress(self,docs:list[dict]):
        chunks = [doc['chunk'] for doc in docs]
        return self.compressor.compress_prompt(
            chunks,
            rate=self.config['compression']['rate'],
            )
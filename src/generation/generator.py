from dotenv import load_dotenv
from openai import OpenAI
import os
from src.retrieval.reranker import ReRanker
from src.generation.compressor import Compressor
load_dotenv()

class Generator:
    def __init__(self):
        self.reranker=ReRanker()
        self.compressor=Compressor()
        self.ENDPOINT=os.environ['PROJECT_ENDPOINT']
        self.API_KEY=os.environ['OPENAI_TOKEN']
        self.DEPLOYMENT_NAME=os.environ['DEPLOYMENT_NAME']
        self.client=OpenAI(base_url=self.ENDPOINT,api_key=self.API_KEY)
        self.SYSTEM_PROMPT="""You are an AI assistant. Answer the user's question using ONLY the information provided in the given context.

- Do NOT use any external knowledge.
- Do NOT make assumptions or add information not present in the context.
- If the answer cannot be found in the context, respond with: "I don't know based on the provided context."
- Keep the answer clear, accurate, and grounded strictly in the context."""
    
    def generate(self,query:str):
        scored_docs=self.reranker.rerank(query=query,top_k=5)
        compressed_context=self.compressor.compress(scored_docs)

        # completion=self.client.chat.completions.create(
        #     model=self.DEPLOYMENT_NAME,
        #     messages=[
        #         {"role": "system", "content": self.SYSTEM_PROMPT},
        #         {"role": "user", "content": f"Context:\n{compressed_context}\n\nQuestion:\n{query}"}
        #     ]
        # )
        print(compressed_context)


    
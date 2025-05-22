# 封装Embedding以在langChain中使用
from typing import List
from langchain_core.embeddings import Embeddings
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

class ArkEmbeddings(Embeddings):

    def __init__(self, api_key: str, api_url: str, model: str):
        """
        实例化client.

        Args:
            api_key (str): 模型key 
            api_url (str): 模型url
            model (str): 模型名
        """


        self.client = OpenAI(
            api_key=api_key,
            base_url=api_url,
        )
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embeddings.

        Args:
            texts (List[str]): 要生成 embedding 的文本列表.
        
        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表. 每个 embedding 都表示为一个浮点值列表.
        """
        result = []
        # 每批处理文本
        for i in range(0, len(texts), 64):
            embeddings = self.client.embeddings.create(
                model=self.model,
                input=texts[i:i+64]
            )
            result.extend([embeddings.embedding for embeddings in embeddings.data])
        
        return result

    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.

        Args:
            texts (str): 要生成 embedding 的文本.

        Return:
            embeddings (List[float]): 输入文本的 embedding, 一个浮点值列表.
        """
        return self.embed_documents([text])[0]

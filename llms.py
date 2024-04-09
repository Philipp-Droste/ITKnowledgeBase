from os import getenv
from abc import ABC, abstractmethod
from typing import (
    Any,
    List,
)
from langchain_community.llms import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class BaseLLM(ABC):
    """LLM wrapper should take in a prompt and return a string."""

    @abstractmethod
    def generate(self, messages: List[str]) -> str:
        """Comment"""

    @abstractmethod
    async def generateStreaming(
        self, messages: List[str], onTokenCallback
    ) -> List[Any]:
        """Comment"""

    @abstractmethod
    async def num_tokens_from_string(
        self,
        string: str,
    ) -> str:
        """Given a string returns the number of tokens the given string consists of"""

    @abstractmethod
    async def max_allowed_token_length(
        self,
    ) -> int:
        """Returns the maximum number of tokens the LLM can handle"""

class Llama2(BaseLLM):
    """LLM class that wraps locally running Ollama llama2 model"""

    def __init__(self) -> None:
        self.llm = ChatOllama(
            temperature=0,
            base_url=getenv("OLLAMA_BASE_URL"),
            model="llama2",
            streaming=True,
            # seed=2,
            top_k=10,  # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
            top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
            num_ctx=3072,  # Sets the size of the context window used to generate the next token.
        )

    def generate(self, messages: List[str]) -> str:
        prompt = ChatPromptTemplate.from_messages([
            (dict["role"], dict["content"]) for dict in messages
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke()

    def generateStreaming(self, messages: List[str], onTokenCallback) -> List[str]:
        prompt = ChatPromptTemplate.from_messages([
            (dict["role"], dict["content"]) for dict in messages
        ])
        chain = prompt | self.llm | StrOutputParser()
        return list(chain.stream())
    
    def num_tokens_from_string(self, string: str) -> int:
        return self.llm.get_num_tokens(string)
    
    def max_allowed_token_length(self) -> int:
        return self.llm.num_ctx
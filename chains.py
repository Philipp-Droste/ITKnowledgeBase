from os import getenv
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

ollama_base_url = getenv("OLLAMA_BASE_URL")

def load_llm():
    llm = ChatOllama(
        temperature=0,
        base_url=ollama_base_url,
        model="llama2",
        streaming=True,
        # seed=2,
        top_k=10,  # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
        top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
        num_ctx=3072,  # Sets the size of the context window used to generate the next token.
    )
    embeddings = OllamaEmbeddings(
        base_url=ollama_base_url, model="llama2"
    )
    dimension = 4096
    return llm, embeddings, dimension
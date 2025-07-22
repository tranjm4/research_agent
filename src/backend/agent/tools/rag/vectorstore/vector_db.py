from tools.rag.vectorstore.shards.vectorstore import load_shards
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()
import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        
def load_db(path: str = "./saved_vectorstore_shards", k=1) -> list:
    """
    Load vector store shards from the specified path.

    Args:
        embedding_module (OpenAIEmbeddings): The embedding module to use for loading the shards.
        path (str): The path to the directory containing the vector store shards.

    Returns:
        list: A list of loaded vector store shards.
    """
    print("Loading vector store shards from path:", path)
    embedding_module = OpenAIEmbeddings(model="text-embedding-3-small")
    print(f"Using embedding model: {embedding_module.model}")
    return load_shards(embedding_module, k=k)


if __name__ == "__main__":
    db = load_db()
    for shard in db:
        print(shard)
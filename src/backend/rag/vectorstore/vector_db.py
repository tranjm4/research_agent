from rag.vectorstore.shards.vectorstore import load_shards
from langchain_openai import OpenAIEmbeddings

from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()
import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class VectorStore:
    """
    A class to represent a vector store for storing and querying documents.
    This class is a wrapper around the vector store shards.
    """

    def __init__(self, shards: list | None = None, k: int = 2):
        """
        Initializes the VectorStore with the provided shards.

        Args:
            shards (list): A list of vector store shards.
        """
        if shards is None:
            self.shards = load_shards(OpenAIEmbeddings(model="text-embedding-3-small"), k=2)
        else:
            self.shards = shards

    def query(self, input: str) -> list:
        """
        Query the vector store with the given input string.

        Args:
            input (str): The input string to query the vector store.

        Returns:
            list: A list of documents that match the query.
        """
        with ThreadPoolExecutor(max_workers=len(self.shards) // 2) as executor:
            futures = [executor.submit(shard.invoke, input) for shard in self.shards]
            query_results = []
            for future in as_completed(futures):
                try:
                    results = future.result()
                    if results:
                        query_results.extend(results)
                except Exception as e:
                    print(f"Error querying shard: {e}")
        return query_results
        
def load_db(path: str = "./saved_vectorstore_shards") -> list:
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
    return load_shards(embedding_module)


if __name__ == "__main__":
    db = load_db()
    for shard in db:
        print(shard)
from rag.vectorstore.shards.vectorstore import load_shards
from langchain_openai import OpenAIEmbeddings

from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()
import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def query(input: str, shards: list | None = None, embedding: OpenAIEmbeddings | None = None) -> list:
    """
    Query the vector database with the given input string.

    Args:
        input (str): The input string to query the vector database.

    Returns:
        list: A list of documents that match the query.
    """
    if shards is None:
        # If shards are not provided, load them from the default path
        shards = load_shards()

    if embedding is None:
        # If embedding is not provided, use the default OpenAI embedding model
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    query_results = []

    with ThreadPoolExecutor(max_workers=len(shards)) as executor:
        futures = []
        for shard in shards:
            # Submit the query task for each shard
            futures.append(executor.submit(shard.invoke, input))

        for future in as_completed(futures):
            try:
                # Collect results from each shard
                results = future.result()
                if results:
                    query_results.append(results)
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
    embedding_module = OpenAIEmbeddings(model="text-embedding-3-small")
    return load_shards(embedding_module)


if __name__ == "__main__":
    db = load_db()
    for shard in db:
        print(shard)
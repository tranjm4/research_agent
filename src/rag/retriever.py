from rag.vectorstore.vector_db import load_db

from concurrent.futures import ThreadPoolExecutor, as_completed

class Retriever:
    """
    A class to handle retrieval of documents from a vector database.
    """

    def __init__(self, decomposition_model: str = "default"):
        """
        Initializes the Retriever with a decomposition model and loads the vector database.
        """
        self.decomposition_model = decomposition_model
        self.shards = load_db()

    def invoke(self, input: str) -> list:
        """
        Query the vector database with the given input string.

        Args:
            input (str): The input string to query the vector database.

        Returns:
            list: A list of documents that match the query.
        """
        query_results = []
        with ThreadPoolExecutor(max_workers=len(self.shards)) as executor:
            futures = [executor.submit(shard.invoke, input) for shard in self.shards]
            for future in as_completed(futures):
                try:
                    results = future.result()
                    if results:
                        query_results.append(results)
                except Exception as e:
                    print(f"Error querying shard: {e}")
        return query_results

if __name__ == "__main__":
    retriever = Retriever()
    print(retriever)
    print(retriever.invoke("What are some recent breakthroughs in AI?"))
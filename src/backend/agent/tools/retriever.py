from agent.tools.rag.vectorstore.vector_db import load_db

from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import CrossEncoder

from langchain_core.tools import tool

from agent.utils import ModelConfig, log_stats

class Retriever:
    """
    A class to handle retrieval of documents from a vector database.
    """
    def __init__(self, **config):
        """
        Initializes the Retriever with a decomposition model and loads the vector database.
        Args:
            config (ModelConfig): Configuration dictionary containing model parameters, metadata, and logging information.
        """
        params = config.get("params", {})
        encoder_model = params.get("encoder_model", None)
        num_per_shard = params.get("num_per_shard", 3)
        top_k = params.get("top_k", 5)
        
        self.shards = load_db()
        if encoder_model is None:
            self.num_search = num_per_shard * len(self.shards)
            self.reranker = None
        else:
            self.reranker = Reranker(model=encoder_model, top_k=top_k)
            self.num_search = top_k
        self.metadata = config.get("metadata", {})
        self.logging = config.get("logs", {})
        
        if not self.logging:
            print("No logging configuration provided. Please check your config file.\n")
            raise ValueError(f"No logging configuration provided. Config: {config}")
        
        self.log_stats = log_stats(self.metadata, self.logging["out_file"], self.logging["err_file"])
        self.query = self.log_stats(self._query)

    def _query(self, prompt: str) -> str:
        """
        Query the vector database with the given input string.

        Args:
            query (dict): The input dictionary to query the vector database.

        Returns:
            str: A string representation of the list of documents that match the query.
        """
        if type(prompt) is not str:
            raise ValueError(f"Input to Retriever must be a string. Received {type(prompt)}")
        
        query_results = []
        with ThreadPoolExecutor(max_workers=len(self.shards)) as executor:
            futures = [executor.submit(shard.invoke, prompt) for shard in self.shards]
            for future in as_completed(futures):
                try:
                    results = future.result()
                    if results:
                        for r in results:
                            query_results.append(r)
                except Exception as e:
                    print(f"Error querying shard: {e}")

        if self.reranker is None:
            return query_results
        else:
            # Rerank the results using the reranker model
            reranked_results = self.reranker.rerank(query_results, prompt)
            parsed_results = self.parse_search_results(reranked_results)
            return parsed_results
        
    def parse_search_results(self, results: list) -> str:
        """
        Parse the search results into a string format for the core model to use.

        Args:
            results (list): The list of documents to parse.

        Returns:
            str: A string representation of the search results.
        """
        return "\n".join([f"Document {i+1}: {doc.page_content} ({doc.metadata['url']})" for i, doc in enumerate(results)]) if results else "No relevant documents found."
    
    def as_tool(self):
        """
        Returns the Retriever as a LangChain tool.

        Returns:
            RunnableLambda: A runnable that performs the retrieval operation.
        """
        @tool(name_or_callable="retriever_tool",
              description="Retrieves relevant documents from a vector database based on the input query.",
              parse_docstring=True)
        def retriever_tool(query: str) -> str:
            """Retrieves potentially relevant documents from a vector database based on the input query.
            
            Args:
                query: 
                    The input query to retrieve documents from the vector database.
            
            Returns:
                A string containing the retrieved documents.
            """
            return self.query(query)

        return retriever_tool

class Reranker:
    """
    A class to handle reranking of retrieved documents.
    """

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 5):
        """
        Initializes the Reranker with a specified model.
        """
        self.model = CrossEncoder(model)
        self.top_k = top_k

    def rerank(self, documents: list, query: str) -> list:
        """
        Rerank the given documents based on their relevance.

        Args:
            documents (list): The list of documents to rerank.
            query (str): The query string used for reranking.

        Returns:
            list: A list of reranked documents.
        """
        pairs = [(query, doc.page_content) for doc in documents]

        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, documents), key= lambda x: x[0], reverse=True)
        
        return [doc for _, doc in ranked[:self.top_k]]

if __name__ == "__main__":
    retriever = Retriever(decomposition_model=None, num_per_shard=3)
    query = {"input": "What is the role of LLMs in K-12 education?"}
    results = retriever.invoke(query)
    reranked_results = retriever.reranker.rerank(results, query["input"])
    
    for doc in reranked_results:
        print(doc.page_content)
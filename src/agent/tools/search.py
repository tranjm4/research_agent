"""
File: agent/tools/search.py

This module defines a search model for the agent, which is used to retrieve relevant information from a vector store.
"""


from rag.prompting import prompt_decomposition
from rag.vectorstore.vector_db import load_db

from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

def invoke(input_prompt: str) -> list:
    """
    Invoke the search model with the given input string.

    Args:
        input (str): The input string to search in the vector store.
        embedding_module (OpenAIEmbeddings, optional): The embedding module to use for the search. Defaults to None.
        path (str): The path to the vector store shards. Defaults to "./saved_vectorstore_shards".

    Returns:
        list: A list of documents that match the search query.
    """
    vector_db = load_db()
    decomposition = prompt_decomposition.invoke(input_prompt)
    
    print(decomposition[0]) # List of keywords and phrases extracted from the input prompt
    
    shard = vector_db[0]
    
    result = shard.invoke(
        input_prompt,
        filter={
            "page_content": {
                "$in": decomposition[0]["keywords"]
            }
        }
    )
    
    print(result)
    return result
    
    # with ThreadPoolExecutor(max_workers=len(vector_db)) as executor:
    #     futures = []
    #     for shard in vector_db:
    #         # Submit the search task for each shard
    #         futures.append(executor.submit(shard.similarity_search, 
    #                                        input_prompt, ))
        
    #     for future in as_completed(futures):
    #         try:
    #             # Collect results from each shard
    #             result = future.result()
    #             if result:
    #                 results.extend(result)
    #         except Exception as e:
    #             print(f"Error querying shard: {e}")
    

if __name__ == "__main__":
    invoke("What papers are related to quantum computing?")
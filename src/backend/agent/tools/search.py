"""
File: agent/tools/search.py

This module defines a search model for the agent, which is used to retrieve relevant information from a vector store.
"""

from agent.tools.wrapper import ModelWrapper
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from agent.tools.rag.prompting import keyword_decomposition
from agent.tools.rag.vectorstore.vector_db import VectorStore

from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

class SearchModel(ModelWrapper):
    """
    A model wrapper for the search functionality, allowing the agent to search through a vector store.
    """
    default_prompt = """
    You are a search expert in various academic domains.
    Your task is to select the most relevant documents from a
    given list of documents based on the provided user input.
    
    You will be given a user prompt, followed by a numbered list of documents, each with a URL and its abstract.
    Your task is to select the top 3 most relevant documents based on the user prompt.
    
    Your job is to return a list of the top 3 document numbers.
    Under any circumstances, do not return more than 3 documents.
    Under any circumstances, do not include any other text or information in your response.
    
    Example input:
    User prompt: "What are the latest advancements in quantum computing and cybersecurity?"
    Documents:
    1. URL: https://example.com/doc1 - Abstract: "This paper discusses the latest advancements in quantum computing, including new algorithms and hardware improvements..."
    2. URL: https://example.com/doc2 - Abstract: "This article reviews the current state of quantum computing research and its implications for the future."
    3. URL: https://example.com/doc3 - Abstract: "This paper presents a comprehensive overview of quantum machine learning techniques and their applications."
    4. URL: https://example.com/doc4 - Abstract: "This study explores the challenges and opportunities in quantum computing, focusing on scalability and error correction."
    5. URL: https://example.com/doc5 - Abstract: "This article provides a detailed analysis of quantum cryptography and its potential impact on secure communications."
    6. URL: https://example.com/doc6 - Abstract: "This paper examines the role of quantum computing in solving complex optimization problems and its applications in various fields."
    7. URL: https://example.com/doc7 - Abstract: "This research explores the intersection of quantum computing and artificial intelligence, highlighting recent breakthroughs and future directions."
    8. URL: https://example.com/doc8 - Abstract: "This paper discusses the implications of quantum computing for cybersecurity and data privacy."
    9. URL: https://example.com/doc9 - Abstract: "This article reviews the latest developments in quantum hardware, including superconducting qubits and ion traps."
    
    Example output:
    [2, 5, 8]
    
    Reasoning (not to be included in the output):
    - The user prompt is focused on advancements in quantum computing and cybersecurity.
    - Document 2 provides a general overview of quantum computing research, which is relevant but less specific than the other two.
    - Document 5 discusses quantum cryptography, which is directly related to cybersecurity.
    - Document 8 discusses the implications of quantum computing for cybersecurity, making it relevant.
    
    Under any circumstances, do not return more than 3 documents.
    Under any circumstances, do not include any other text or information in your response.
    """

    def __init__(self, system_prompt: str, model: str = "llama3.2:3b", **kwargs):
        self.input_template = lambda x: {
            "input": x["input"]
        }
        self.vector_db = VectorStore(k=1)
        
        self.parse_func = RunnableLambda(lambda x: x)
        

        super().__init__(
            agent_type="search",
            system_prompt=system_prompt,
            input_template=self.input_template,
            model=model,
            parse_func=self.parse_func,
            **kwargs
        )
        
        self.prompt_template = self.build_prompt_template(SearchModel.default_prompt)
        
        self.chain = RunnableLambda(lambda x: x["input"]) \
            | RunnableLambda(lambda x: self.search_db(x)) \
            | RunnableLambda(lambda x: self.parse_search_results(x)) \
            | RunnableLambda(lambda x: self.build_chat_prompt(x)) \
            | RunnableLambda(lambda x: self.run_model(x)) \
            | RunnableLambda(lambda x: self.log_stats(x)) \
            | RunnableLambda(lambda x: self.parse_model_output(x))
            
    def search_db(self, query: str) -> dict:
        """
        Searches the vector store for relevant documents based on the user query.
        
        Args:
            query (str): The user input to search for in the vector store.
        
        Returns:
            dict: A dictionary containing the user input and the search results,
            with keys:
            {
                "user_input": str,
                "search_results": list[Document]
            }
                which will be passed through the chain.
        """
        
        print(f"Querying vector store with input: {query}")
        results = self.vector_db.query(query)
        return {
            "user_input": query,
            "search_results": results
        }
    
    def parse_search_results(self, x: dict) -> dict:
        """
        Parses the search results to extract relevant information.

        Args:
            x (dict): A dictionary containing the passed variables in the chain
            with keys:
                - "user_input" (str): The user input string.
                - "search_results" (list[Document]): A list of Document objects containing the search results.

        Returns:
            dict: {
                "user_input": str,
                "search_results": list[Document],
                "parsed_search_results": str
            }
        """
        search_results = x["search_results"]
        
        x["parsed_search_results"] = "\n".join(
            [f"{i+1}. URL: {doc.metadata['url']} - Abstract: {doc.page_content}" for i, doc in enumerate(search_results)]
        )
        
        return x

    def build_chat_prompt(self, x: str) -> dict:
        """
        Builds the chat prompt for the search model.
        
        Args:
            context (str): The input data containing the context
        
        Returns:
            dict: {
                "user_input": str,
                "search_results": list[Document],
                "parsed_search_results": str,
                "prompt": str
            }
        """
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(SearchModel.default_prompt),
                HumanMessagePromptTemplate.from_template("User Prompt: {user_input}\n"
                                                         "Documents: \n{input}"),
            ]
        )
        
        prompt = chat_prompt_template.invoke(
            {
                "user_input": x["user_input"],
                "input": x["parsed_search_results"]
            }
        )
        
        x["prompt"] = prompt
        return x
    
    def run_model(self, x: dict) -> dict:
        """
        Runs the search model on the provided input.
        Args:
            x (dict): A dictionary containing the input data with keys:
                - "user_input" (str): The user input string.
                - "search_results" (list[Document]): A list of Document objects containing the search results.
                - "parsed_search_results" (str): The parsed search results string.
                - "prompt" (str): The chat prompt to be sent to the model.
            
        Returns:
            dict: The output from the model, which will be passed through the chain.
            {
                "user_input": str
                "search_results": list[Document],
                "parsed_search_results": str,
                "prompt": str,
                "model_output": str
            }
        """
        
        output = self.model.invoke(x["prompt"])
        
        x["model_output"] = output
        print(output)
        return x
    
    def log_stats(self, x: dict) -> dict:
        """
        Logs the statistics of the search model output.
        
        Args:
            x (dict): A dictionary containing the input data with keys:
                - "user_input" (str): The user input string.
                - "search_results" (list[Document]): A list of Document objects containing the search results.
                - "parsed_search_results" (str): The parsed search results string.
                - "prompt" (str): The chat prompt to be sent to the model.
                - "model_output" (OllamaChatCompletion): The output from the model.
        
        Returns:
            dict: The input data with an additional key for logging purposes.
        """
        super().log_stats(x["model_output"])
        return x

    def parse_model_output(self, x: dict) -> list[int]:
        """
        Parses the model output to extract the top 3 document numbers.
        
        Args:
            model_output (str): The output from the model.
        
        Returns:
            list[int]: A list of the top 3 document numbers.
        """
        try:
            # Extract the document numbers from the model output
            doc_numbers = [int(num)-1 for num in x["model_output"].content.strip().strip("[]").split(",")]
            
            documents_list = x["search_results"]
            
            parsed_model_output = []
            for num in doc_numbers:
                parsed_model_output.append(documents_list[num])
                
            return parsed_model_output
                
        except ValueError as e:
            print(f"Error parsing model output: {e}")
            return []
        
if __name__ == "__main__":
    # search_model = SearchModel(system_prompt=SearchModel.default_prompt, model_name="llama3.2:3b")
    # input_prompt = {
    #     "input": "What are the latest advancements in large language models and their applications in education?"
    # }
    
    # result = search_model.invoke(input_prompt)
    # print("Search Results:", result)
    search_model = SearchModel(system_prompt=SearchModel.default_prompt, 
                               version_name="search/mistal:7b",
                               model="mistral:7b", 
                               num_ctx=20000, 
                               temperature=0.1)
    input_prompt = {
        "input": "What are language models' applications in education?"
    }

    result = search_model.invoke(input_prompt)
    for r in result:
        print()
        print(f"Document URL: {r.metadata['url']}")
        print()
        print(f"Document Abstract: {r.page_content}\n")
        print("---"*20)
"""
File: src/backend/agent/eval/search/precision_eval_model.py

This module details a model for evaluating the precision of a search system.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

import json

class PrecisionEvalModel:
    """
    A model for evaluating the precision of a search system.
    
    Attributes:
        model (ChatOllama): The language model used for evaluation.
        prompt_template (PromptTemplate): The template for generating prompts.
    """
    
    def __init__(self, model_name: str = "llama3.2:1b", temperature: float = 0.1):
        """
        Initializes the PrecisionEvalModel with a specified model and temperature.
        
        Args:
            model_name (str): The name of the model to use for evaluation.
            temperature (float): The temperature setting for the model.
        """
        self.model = ChatOllama(model=model_name, temperature=temperature)
        self.prompt = """
        You are an expert in comparing search results.
        Given a search query and a document's text, determine if the document is relevant or related to the query.

        Under all circumstances, you must respond with a JSON object in the following format:
        {{"result": <0 or 1>}}
        Do not include any additional text or explanations.
        Do not include any newlines after the brackets.
        
        Example:
        Query: "What is the capital of France?"
        Document: "The capital of France is Paris."
        Response: {{"result": 1}}
        
        Query: {query}
        Document: {document}
        Response: 
        """
        
        self.prompt_template = PromptTemplate(
            input_variables=["query", "document"],
            template=self.prompt
        )
    def evaluate(self, query: str, document: str) -> dict:
        """
        Evaluates the relevance of a document to a search query.
        """
        prompt = self.prompt_template.format(query=query, document=document)
        response = self.model.invoke(prompt)
        response = json.loads(response.content)
        return response
    
if __name__ == "__main__":
    # Example usage
    eval_model = PrecisionEvalModel(model_name="llama3.2:3b", temperature=0)
    query = "What are recent advancements in cell research?"
    document = """
    With the recent developments in large language models (LLMs) and their widespread availability through open source models and/or low-cost APIs, several exciting products and applications are emerging, many of which are in the field of STEM educational technology for K-12 and university students. There is a need to evaluate these powerful language models on several benchmarks, in order to understand their risks and limitations. In this short paper, we summarize and analyze the performance of Bard, a popular LLM-based conversational service made available by Google, on the standardized Physics GRE examination.\n9. URL: https://arxiv.org/abs/2408.07144 - Abstract: This chapter critically examines the potential contributions of modern language models to theoretical linguistics. Despite their focus on engineering goals, these models' ability to acquire sophisticated linguistic knowledge from mere exposure to data warrants a careful reassessment of their relevance to linguistic theory. I review a growing body of empirical evidence suggesting that language models can learn hierarchical syntactic structure and exhibit sensitivity to various linguistic phenomena, even when trained on developmentally plausible amounts of data. While the competence/performance distinction has been invoked to dismiss the relevance of such models to linguistic theory, I argue that this assessment may be premature. By carefully controlling learning conditions and making use of causal intervention methods, experiments with language models can potentially constrain hypotheses about language acquisition and competence. I conclude that closer collaboration between theoretical linguists and computational researchers could yield valuable insights, particularly in advancing debates about linguistic nativism.
    """

    result = eval_model.evaluate(query, document)
    print(result.content)  # Expected output: {"result": 1} if relevant, otherwise {"result": 0}
    # convert the result to a dictionary
    result_dict = json.loads(result.content)
    print(result_dict)
    print(type(result_dict))
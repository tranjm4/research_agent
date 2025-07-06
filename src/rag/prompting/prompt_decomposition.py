"""
File: src/rag/prompt_decomposition.py

This module provides functionality to decompose prompts into
keywords and phrases, which can be used as filters prior to querying
the vector database.

This module also provides functionality to rephrase the original prompt
into a more useful prompt for the vector database query.
"""

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from random import sample

from prompt_synthesis import load_synthesis_prompts
import ast

from pprint import pprint


def get_decomposition_prompt():
    """
    Returns the decomposition prompt template.
    """
    decomposition_prompt = """INSTRUCTIONS:
    You are an expert in research papers and prompt decomposition.
    Your task is to decompose the input prompt into keywords and phrases.
    
    You will receive a series of input prompts that are related to research papers.
    For each input prompt, return a list of keywords and phrases that can be used to filter
    the vector database query for relevant research papers.

    For each prompt, produce the output in the following Python dict format, separated by new lines:
    {{"keywords": ["keyword1", "keyword2", "keyword3"]}}
    {{"keywords": ["keyword1", "keyword2", "keyword3", "additional keyword"]}}
    
    Do not include any additional text or explanations.
    
    EXAMPLES

    INPUT:
    What are recent breakthroughs in using diffusion models for image generation?
    I'm looking for papers on quantum computing and its applications in cryptography.
    
    OUTPUT:
    {{"keywords": ["diffusion models", "image_generation", "recent breakthrough"]}}\n
    {{"keywords": ["quantum computing", "cryptography", "applications", "applications in cryptography"]}}\n
    """
    
    return decomposition_prompt

def parse_decomposition_output(content):
    """
    Parses the output from the decomposition prompt to extract keywords and phrases.
    
    Args:
        output (str): The output string from the LLM containing keywords and phrases.
        
    Returns:
        dict: A dictionary containing the keywords and phrases.
    """
    output_list = content.strip().split("\n")
    output_dict = [ast.literal_eval(line) for line in output_list if line.strip()]
    return output_dict

def compute_decomposition(num_samples=5):
    """
    Computes the decomposition of the input prompt into keywords and phrases.
    
    Args:
        num_samples (int): The number of samples to return.
        
    Returns:
        dict: A dictionary containing the keywords and phrases.
    """
    decomposition_prompt = get_decomposition_prompt()
    
    prompts = load_synthesis_prompts()
    sampled_prompts = sample(prompts, num_samples)
    sampled_prompts_str = "\n".join(sampled_prompts)
    
    system_template = SystemMessagePromptTemplate.from_template(decomposition_prompt)
    human_template = HumanMessagePromptTemplate.from_template("{input_prompt}")

    llm = ChatOllama(model="llama3.2", temperature=0.1, max_tokens=1000)

    decomposition_template = ChatPromptTemplate.from_messages(
        [system_template, human_template]
    )

    chain = RunnableSequence(
        {
            "input_prompt": lambda x: x["input_prompt"]
        },
        decomposition_template | llm \
        | RunnableLambda(lambda x: parse_decomposition_output(x.content))
    )
    
    keywords_list = chain.invoke({"input_prompt": sampled_prompts_str})
    
    zipped_results = zip(sampled_prompts, keywords_list)
    for r in zipped_results:
        prompt, output = r
        print(f"Prompt: {prompt}\nDecomposed Output: {output}\n")

    return keywords_list
    


if __name__ == "__main__":
    compute_decomposition(5)

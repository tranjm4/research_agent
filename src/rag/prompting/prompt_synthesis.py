"""
File: src/rag/synthesis/prompt_synthesis.py

This module details the design of data synthesis for generating prompts
for querying a vector database.

It includes the design of an LLM that can generate new prompts based on existing ones,

"""

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from random import sample
import regex as re

from argparse import ArgumentParser
from tqdm import tqdm


def create_synthesis_prompt():
    synthesis_prompt = """INSTRUCTIONS:
    You are an expert in synthesizing prompts for querying vector databases.
    Your task is to generate new prompts based on the input prompt that can be used to query the vector database.
    You will return {n} synthesized prompts that are relevant but varying in wording and structure.
    Keep the prompts concise, no more than 15 words each.
    You will produce prompts that are relevant to the input prompts but also introduce new topics or perspectives.
    
    Provide each of your new prompts on a new line and number them sequentially.
    
    You are encouraged to generate prompts about subjects that are not related to the given input prompts.

    Topics can include, but are not limited to, astrophysics, general relativity, quantum cosmology, high energy physics, mathematical physics,
    nonlinear sciences, nuclear theory, physics, geophysics, quantum physics, mathematics (number theory, topology, ring and group theory, etc.),
    computer architecture, machine learning, artificial intelligent, cryptography, operating systems, automata theory, cybersecurity,
    computer vision, quantitative biology, quantitative finance, econonics, statistical theory, electrical engineering.

    You will not include any additional text or explanations.

    EXAMPLE:
    
    INPUT:
    prompt 1
    prompt 2
    prompt 3
    
    OUTPUT:
    1. new prompt 1
    2. new prompt 2
    3. new prompt 3
    4. new prompt 4
    5. new prompt 5
    ...
    """
    
    return synthesis_prompt

def parse_synthesis_output(output):
    """
    Parses the output from the synthesis prompt to extract synthesized prompts.

    Args:
        output (str): The output string from the LLM containing synthesized prompts.
    
    Returns:
        list: A list of synthesized prompts.
    """
    
    # Split the output by new lines and filter out empty lines
    prompt = output.strip()
    if not prompt:
        return None

    regex = r"^(\d+\. )(.+)$"
    # Filter prompts that match the expected numbered format
    prompt = re.match(regex, prompt).group(2) if re.match(regex, prompt) else None
    
    return prompt

def save_synthesis_prompt(prompt):
    """
    Writes a synthesized prompt to a new line in a file.

    Args:
        prompt (str): The synthesized prompt to save.
    """
    with open("synthesis_prompts.txt", "a") as file:
        file.writelines(f"{prompt}\n")

def load_synthesis_prompts():
    try:
        with open("synthesis_prompts.txt", "r") as file:
            return [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        return create_synthesis_prompt()
    
def get_sample_prompts(n=10):
    """
    Returns a subsample of prompts for constructing the system prompt.

    Args:
        n (int): Number of prompts to return. Defaults to 5.
    Returns:
        list: A list of sample prompts.
    """
    
    prompts = load_synthesis_prompts()
    return sample(prompts, n)
    
def get_synthesis_chain(n=5, temperature=0.5):
    """
    Constructs a synthesis chain that generates prompts based on the input prompt.

    Args:
        n (int): Number of synthesized prompts to generate. Defaults to 5.
    Returns:
        Runnable: A runnable that generates synthesized prompts.
    """
    
    synthesis_prompt = create_synthesis_prompt()
    system_template = SystemMessagePromptTemplate.from_template(synthesis_prompt)
    human_template = HumanMessagePromptTemplate.from_template("{input_prompts}")
    
    llm = ChatOllama(model="llama3.2", temperature=temperature, max_tokens=20000, top_p=0.9, top_k=40)
    
    prompt_template = ChatPromptTemplate(
        messages=[system_template, human_template],
        input_variables=["input_prompts"]
    )
    
    synthesis_chain = RunnableSequence(
        {
            "n": RunnableLambda(lambda x: x),
            "input_prompts": RunnableLambda(lambda x: x["input_prompts"])
        },
        prompt_template | llm
    )
    
    return synthesis_chain

def collect_synthesis_prompts(n=5, temperature=0.5):
    """
    Collects synthesized prompts from the user and saves them to a file.
    """
    chain = get_synthesis_chain(n=n, temperature=temperature)
    example_prompts = get_sample_prompts()
    example_prompts_str = "\n".join([prompt for prompt in example_prompts])

    result = chain.invoke({
        "input_prompts": example_prompts_str
    })

    for prompt in result.content.split("\n"):
        parsed_prompt = parse_synthesis_output(prompt)
        save_synthesis_prompt(parsed_prompt) if parsed_prompt is not None else None


if __name__ == "__main__":
    parser = ArgumentParser(description="Collect synthesized prompts for querying vector databases.")
    parser.add_argument("--n", type=int, default=5, help="Number of synthesized prompts to generate.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for the LLM generation.")
    parser.add_argument("--iter", type=int, default=1, help="Number of iterations to perform synthesis")
    args = parser.parse_args()

    for i in tqdm(range(args.iter), desc="Synthesizing prompts..."):
        collect_synthesis_prompts(n=args.n, temperature=args.temperature)
    # Uncomment the line below to run the synthesis chain and generate prompts

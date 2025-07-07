from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from rag.prompting import prompt_decomposition as decomposition


def core_prompt():
    """
    Returns the core prompt template for the agent.
    """
    system_template = """
    You are a research assistant. Your task is to assist with research tasks by providing relevant information and insights.
    You will receive a series of input prompts that are related to research papers.
    
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are a research assistant. Your task is to assist with research tasks by providing relevant information and insights."
            ),
            HumanMessagePromptTemplate.from_template(
                "{input}"
            )
        ]
    )

    return prompt
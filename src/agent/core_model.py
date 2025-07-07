from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from rag.prompting import prompt_decomposition as decomposition

class CoreModel:
    """
    CoreModel class that initializes the core model for the agent.
    It sets up the model, prompt, and runnable sequence for processing input.
    """

    def __init__(self, prompt_template: str, model_name: str = "llama3.2:3b", **kwargs):
        """
        Initializes the CoreModel with a prompt template and model name.
        Args:
            prompt_template (str): The template for the prompt.
            model_name (str): The name of the model to use.
            kwargs: Additional keyword arguments for model configuration.
        """
        self.model = ChatOllama(model_name=model_name,
                                **kwargs)  # Initialize the model with the given name and additional parameters
        self.prompt_template = prompt_template
        self.runnable = RunnableSequence(
            {
                "input": lambda x: x["input"],
                "question_trace": lambda x: x["question_trace"],
                "answer_trace": lambda x: x["answer_trace"],
            },
            # include planner before sending it to the model
            self.prompt_template | self.model
        )


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
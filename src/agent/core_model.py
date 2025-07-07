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
        self.chain = RunnableSequence(
            {
                "input": lambda x: x["input"],
                "question_trace": lambda x: x["question_trace"],
                "answer_trace": lambda x: x["answer_trace"],
            },
            # include planner before sending it to the model
            self.prompt_template | self.model
        )
        
        """
        IDEA: consider making planner model create a runnable sequence chain
        """
    def invoke(self, input_data: dict):
        """
        Invokes the core model with the provided input data.
        Args:
            input_data (dict): A dictionary containing the input data for the model.
        Returns:
            The output from the model after processing the input.
        """
        return self.chain.invoke(input_data)
        
def get_core_prompt():
    """
    Base system prompt for the core model (for prototyping purposes).
    Returns:
        str: The system prompt for the core model.
    """
    prompt = """
    You are a research agent. Your task is to answer user questions and queries.
    You also have access to a set of tools that can help you answer questions.
    
    You will receive a user question and a trace of previous questions and answers.
    Use this information to generate a response that addresses the user's query.
    
    Ensure that your response is clear, concise, and informative.
    """
    
    return prompt
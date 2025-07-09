from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

from rag.prompting import keyword_decomposition as decomposition

import agent.planner as planner
import agent.tools.task_decomposition as task_decomp

from argparse import ArgumentParser

class CoreModel:
    """
    CoreModel class that initializes the core model for the agent.
    It sets up the model, prompt, and runnable sequence for processing input.
    """

    def __init__(self, prompt_template: str, 
                 model: str = "llama3.2:3b", 
                 planner_model: str = "llama3.2:1b",
                 decomp_model: str = "mistral:7b",
                 **kwargs):
        """
        Initializes the CoreModel with a prompt template and model name.
        Args:
            prompt_template (str): The template for the prompt.
            model_name (str): The name of the model to use.
            kwargs: Additional keyword arguments for model configuration.
        """
        self.model = ChatOllama(model=model,
                                **kwargs)  # Initialize the model with the given name and additional parameters
        
        self.planner_model = planner.PlannerModel(
            system_prompt=planner.default_planner_prompt(),
            model_name=planner_model,
            temperature=kwargs.get("temperature", 0.1),
            num_ctx=kwargs.get("max_tokens", 20000),
        )
        self.decomp_model = task_decomp.TaskDecompositionModel(
            system_prompt=task_decomp.decomp_prompt,
            model=decomp_model,
            temperature=kwargs.get("temperature", 0.1),
            num_ctx=kwargs.get("max_tokens", 20000),
        )
        
        self.prompt_template = prompt_template
        self.chain = RunnableSequence(
            {
                "input": lambda x: x["input"],
                "question_trace": lambda x: x["question_trace"],
                "answer_trace": lambda x: x["answer_trace"],
            },
            # include planner before sending it to the model
            self.prompt_template | self.decomp_model \
            | RunnableLambda(lambda subqs: list(map(self.planner_model.invoke, subqs))) \
            |self.model
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
    
    def stream(self, input_data:dict):
        """
        Streams the output from the core model based on the provided input data.
        Args:
            input_data (dict): A dictionary containing the input data for the model.
        Returns:
            A generator that yields chunks of output from the model.
        """
        return self.chain.stream(input_data)
        
def get_core_prompt():
    """
    Base system prompt for the core model (for prototyping purposes).
    Returns:
        str: The system prompt for the core model.
    """
    prompt = """
    You are a research agent. Your task is to answer user questions and queries.
    You also have helper models that provide additional context and information,
    such as summarization, search, and decomposition.
    
    You will be provided with a user's question followed by at least one of the following:
    - a summary of the context
    - a set of potentially relevant documents
    - a decomposition of the question into sub-questions
    
    You should use these to answer the question to the best of your ability.

    Answer succinctly and directly, using the provided context and information.
    If you do not know the answer, say "I don't know" or "I cannot answer that question".
    """
    
    return prompt

def main_loop(args):
    """
    Main loop for the core model.
    Args:
        args: Command line arguments for the model configuration.
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(args.prompt_template),
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessagePromptTemplate.from_template("{context}")
        ]
    )
    # Initialize the core model with the provided arguments
    # prompt_decomposition_model = task_decomp.TaskDecompositionModel(
    #     system_prompt=task_decomp.decomp_prompt,
    #     model=args.decomp_model,
    #     temperature=args.temperature,
    #     num_ctx=args.max_tokens,
    # )
    
    # planner_model = planner.PlannerModel(
    #     system_prompt=planner.default_planner_prompt(),
    #     model_name=args.planner_model,
    #     temperature=args.temperature,
    #     num_ctx=args.max_tokens,
    # )
    
    core_model = CoreModel(
        prompt_template=prompt_template,
        model=args.model,
        planner_model=args.planner_model,
        decomp_model=args.decomp_model,
        temperature=args.temperature,
        num_ctx=args.max_tokens,
    )
    
    question_trace = []
    answer_trace = []
    
    while True:
        # Get user input
        print()
        user_input = input(">>> ")
        if user_input.lower() == 'exit':
            break
        
        # Prepare the input data for the model
        input_data = {
            "input": user_input,
            "question_trace": question_trace,
            "answer_trace": answer_trace
        }
    
        # Invoke the core model and print the output
        output = core_model.stream(input_data)
        print()
        for chunk in output:
            if chunk is not None:
                print(chunk.content, end='', flush=True)
        print()
        
        # Update the traces with the new question and answer
        question_trace.append(user_input)
        answer_trace.append(output)

if __name__ == "__main__":
    # parse args
    parser = ArgumentParser(description="Core Model for Research Agent")
    parser.add_argument("--core_model", type=str, default="llama3.2:3b",
                        help="Name of the model to use (default: llama3.2:3b)")
    parser.add_argument("--planner_model", type=str, default="llama3.2:1b",
                        help="Name of the planner model to use (default: llama3.2:1b)")
    parser.add_argument("--decomp_model", type=str, default="mistral:7b",
                        help="Name of the task decomposition model to use (default: mistral:7b)")
    parser.add_argument("--prompt_template", type=str, default=get_core_prompt(),
                        help="Prompt template for the model (default: base system prompt)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for the model (default: 0.1)")
    parser.add_argument("--max_tokens", type=int, default=20000,
                        help="Maximum tokens for the model (default: 20000)")
    args = parser.parse_args()

    main_loop(args)
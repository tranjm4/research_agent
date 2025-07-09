from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda

from agent.tools.wrapper import ModelWrapper
import agent.planner as planner

import regex as re

class TaskDecompositionModel(ModelWrapper):
    def __init__(self, system_prompt: str, version_name: str, model: str, **kwargs):
        self.input_template = lambda x: {
            "input": x["input"]
        }
        self.parse_func = parse_decomp_output
        self.version_name = version_name

        super().__init__(
            agent_type="task_decomposition",
            version_name=version_name,
            system_prompt=system_prompt,
            input_template=self.input_template,
            model=model,
            parse_func=self.parse_func,
            **kwargs
        )
        
        # Temporary for prototyping decomposition
        self.planner_model = planner.PlannerModel(
            system_prompt=planner.default_planner_prompt(),
            version_name=self.version_name,
            model_name=kwargs.get("model", "mistral:7b"),
            temperature=kwargs.get("temperature", 0.1),
            num_ctx=kwargs.get("max_tokens", 20000),
        )
        
        self.chain = self.chain | RunnableLambda(lambda x: self.prepare_inputs_for_planner(x)) \
            | RunnableLambda(lambda x: list(map(self.planner_model.invoke, x)))
            
    def prepare_inputs_for_planner(self, x: list[str]):
        return [{"input": i} for i in x]

def parse_decomp_output(output):
    """
    Parses the output from the task decomposition model.
    
    Args:
        output (str): The output string from the model.
        
    Returns:
        list: A list of sub-questions extracted from the output.
    """
    output_lines = output.content.strip().split("\n")
    matches = [re.match(r"^(\d+\. )(.*)", line) for line in output_lines]
    return [match.group(2) for match in matches if match]


decomp_prompt = """
INSTRUCTIONS
You are a domain expert. Your task is to break down a complex task into simpler sub-parts.
Given a user question, decompose it into a brief list of sub-questions that can be answered independently.

ANSWER FORMAT
Answer in a numbered list of sub-questions
Do not add any additional information, just the list of sub-questions.
Do not use any special characters or formatting.

EXAMPLE
question: Which country is larger in population: France or Germany?
answer:
1. What is the population of France?
2. What is the population of Germany?
3. Is France larger in population than Germany?

question: How do I get a software engineering job as a student?
answer:
1. How can I make myself a competitive candidate?
2. What are the best resources for making myself a competitive candidate?

question: How has cancer research evolved over the past decade, and what are the current trends in treatment and prevention?
answer:
1. What are some major advancements in cancer research over the past decade?
2. What are the current trends in cancer treatment and prevention?

question: What is the history of quantum computing, and how does it relate to modern research in the field?
answer:
1. What is the history of quantum computing?
2. What does modern research in quantum computing focus on?
3. Are there any differences between the history of quantum computing and modern research?

question:

Use as few sub-questions as possible. Keep the sub-questions concise and focused on the main aspects of the task.
Do not include any additional information or explanations in your response.
"""

if __name__ == "__main__":
    # Example usage
    model = TaskDecompositionModel(system_prompt=decomp_prompt, 
                                   version_name="prototype",
                                   model="llama3.2:3b", 
                                   temperature=0.1,
                                   num_ctx=1000, 
                                   streaming=True)
    print()
    while True:
        user_input = input(">>>  ")
        if user_input.lower() in ["exit", "quit"]:
            break
        input_data = {"input": user_input}
        output = model.invoke(input_data)
        print()
    # input_data = {"input": "How does computer vision research relate to artificial intelligence?"}
    # output = model.invoke(input_data)
    # for result in [x.content.strip() for x in output]:
    #     print(result)
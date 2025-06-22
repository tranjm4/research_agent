import langchain
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from memory import MemoryModule

memory_planner_template = """GENERAL INSTRUCTIONS
Your task is to answer questions. If you cannot answer the question, request a helper or a tool.
If you can answer the question with contextual information, you should not request a helper or a tool.
Fill with Nil where no tool or helper is required

AVAILABLE TOOLS
- Search Tool: Used to search the internet for information
- Math Tool: Used for any mathematical calculations
- Summary Tool: Used to summarize large chunks of text

AVAILABLE HELPERS
- Decomposition: Breaks Complex Questions down into simpler parts

CONTEXTUAL INFORMATION
- {{question_trace}}
- {{answer_trace}}

QUESTION
{{question}}

ANSWER FORMAT
{{"tool":"<FILL>", "helper":"<FILL>"}}

EXAMPLE
question: What is the capital of France?
answer:
{{
  "tool": "Search Tool",
  "helper": "Nil"
}}

question: What is the difference in population between France and Germany?
answer:
{{
  "tool": "Math Tool",
  "helper": "Decomposition"
}}
reasoning: [TODO]

question: What research papers should I read on reinforcement learning?
answer:
{{
  "tool": "Search Tool",
  "helper": "Nil"
}}
"""

# TODO: fill TODO above

class MemoryPlannerChain:
  def __init__(self, template, llm, capacity):
    self.memory = MemoryModule(capacity=capacity)
    self.template = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            ("human", "{question}")
        ]
    )
    self.llm = llm
    self.chain = RunnableSequence(
        {
            "question": lambda x: x["question"],
            "question_trace": lambda x: "\n".join(self.memory.question_trace),
            "answer_trace": lambda x: "\n".join(self.memory.answer_trace),
        },
        self.template | llm
    )

  def invoke(self, input):
    output = self.chain.invoke(input)
    self.add_answer_to_memory(output) # TODO: replace add_answer to actual answer.
    # Right now, answer is just the sequence of requested tools
    self.add_question_to_memory(input["question"])
    return output

  def add_answer_to_memory(self, answer):
    self.memory.add_answer(answer)

  def add_question_to_memory(self, question):
    self.memory.add_question(question)

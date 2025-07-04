import langchain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import OpenAI

decomp_prompt = """INSTRUCTIONS
You are a domain expert. Your task is to break down a complex task into simpler sub-parts

USER QUESTION
{{user_question}}

ANSWER FORMAT
Answer in a numbered list of sub-questions

EXAMPLE
question: Which country is larger in population: France or Germany?
answer:
1. What is the population of France?
2. What is the population of Germany?
3. Is France larger in population than Germany?

question: How do I get a software engineering job as a student?
answer:
1. How do I improve my skills as a programmer?
2. How can I grow my professional network?
3. What kinds of projects are good for my portfolio?
4. How can I prepare for interviews?
5. How can I make myself stand out from other candidates?
"""
decomp_template = ChatPromptTemplate.from_messages(
    [
        ("system", decomp_prompt),
        ("human", "{user_question}")
    ]
)
planner_llm = OpenAI(temperature=0)
planner_chain = RunnableSequence(
    {
        "user_question": lambda x: x["question"],
    },
    decomp_template | planner_llm
)
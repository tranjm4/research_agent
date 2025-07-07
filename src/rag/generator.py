from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


class Generator:
    """
    A class to handle the generation of responses based on retrieved documents.
    """
    allowed_models = [
        "llama3.2:1b",
        "llama3.2:3b",
        "mistral:7b",
        "deepseek-r1:1.5b",
        "deepseek-r1:7b",
        "deepseek-r1:8b"
    ]
    def __init__(self, system_template, model: str = "llama3.2:1b", decomposition_model: str = "default",
                 temperature: float = 0.1, top_k: int = 40, top_p: float = 0.95, max_tokens: int = 10000):
        """
        Initializes the Generator with a decomposition model.
        """
        self.decomposition_model = decomposition_model
        self.system_template = system_template
        
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.system_template),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )
        
        if model not in self.allowed_models:
            raise ValueError(f"Model {model} is not supported. Choose from {self.allowed_models}.")
        
        self.model = ChatOllama(model=model, 
                                temperature=temperature, 
                                top_p=top_p, 
                                top_k=top_k,
                                num_predict=max_tokens
        )
        
        self.chain = self.build_chain()
        
    def build_chain(self):
        """
        Builds the chain for generating responses using the provided system template.
        """
        return RunnableSequence(
            {
                "input": RunnableLambda(lambda x: x["input"]),
                "context": RunnableLambda(lambda x: x["context"]),
            },
            self.prompt_template | self.model
        )

    def invoke(self, user_prompt: str, retrieved_docs: list) -> str:
        """
        Generates a response based on the input and retrieved documents.

        Args:
            input (str): The input query or prompt.
            retrieved_docs (list): A list of documents retrieved from the database.

        Returns:
            str: The generated response.
        """
        
        model_input = {
            "input": user_prompt,
            "context": retrieved_docs
        }
        response = self.chain.invoke(model_input)
        
        return response
    
if __name__ == "__main__":
    system_template = """
    You are a research assistant. Your task is to assist with research tasks
    by providing relevant information and insights.
    You will receive a question and a list of retrieved documents.
    Use the information in the documents to answer the question as accurately as possible.
    
    If the documents do not contain enough information,
    you can use your general knowledge to provide a comprehensive answer.

    You should always cite the documents' URLs you used to answer the question.

    If you cannot find any relevant information in the documents,
    you should respond with "I don't know" or "I cannot answer that question based on the provided documents."
    
    Your response should be concise and to the point.
    
    You will receive the user's question and the retrieved documents in the following format:
    {"input": "<user's question>", "context": ["<doc1>", "<doc2>", "<doc3>"]}
    
    Please provide your response in the following format:
    {"response": "<your response>"}
    """
    
    generator = Generator(system_template=system_template, model="llama3.2:1b")
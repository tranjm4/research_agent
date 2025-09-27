from typing import Literal, Union, Optional, Generator
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable

from utils import logger

class OpenAIParams(BaseModel):
    model_provider: Literal["openai"]
    max_tokens: int
    temperature: float
    timeout: Optional[int]
    max_retries: int

class OllamaParams(BaseModel):
    model_provider: Literal["ollama"]
    model: str
    num_predict: int
    temperature: float
    num_ctx: int

class ModelConfig(BaseModel):
    params: Union[OpenAIParams, OllamaParams] = Field(discriminator="model_provider")

class Model:
    """
    Default class for LLM interfacing
    """
    def __init__(self, config: ModelConfig):
        self.model = self._init_model(config)
        self.metadata = None
        
    def _init_model(self, config: ModelConfig):
        """
        Given the model configuration, creates the model
        """
        # Determine the type of model based on params type
        if isinstance(config.params, OllamaParams):
            model = self._init_ollama(config.params)
        elif isinstance(config.params, OpenAIParams):
            model = self._init_openai(config.params)
        else:
            raise ValueError("Expected 'openai' or 'ollama' as model_provider value")
        
        return model
            
    def _init_openai(self, params: OpenAIParams) -> ChatOpenAI:
        """Initializes the OpenAI model given the config"""
        param_dict = params.model_dump()
        param_dict.pop("model_provider")
        
        model = ChatOpenAI(**param_dict)
        return model
    
    def _init_ollama(self, params: OllamaParams) -> ChatOllama:
        """Initializes the Ollama model given the config"""
        param_dict = params.model_dump()
        param_dict.pop("model_provider")
        
        model = ChatOllama(**param_dict)
        return model
        
    def invoke(self, query: str) -> Runnable:
        return self.model.invoke(input=query)

    
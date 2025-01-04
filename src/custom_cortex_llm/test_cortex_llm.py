from snowflake.snowpark.session import Session
from dotenv import load_dotenv
import os
import litellm
from litellm import CustomLLM
from litellm.types.utils import ModelResponse
import time
from crewai import LLM
from typing import Optional, List, Union, Any, Dict
from langchain.callbacks.manager import CallbackManagerForLLMRun

load_dotenv()

# Snowflake connection setup
connection_params = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_USER_PASSWORD"),
    "role": os.getenv("SNOWFLAKE_ROLE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE")
}

class SnowflakeCortexLLM(CustomLLM):
    """Custom LiteLLM implementation for Snowflake's Cortex"""
    
    def __init__(self, 
                 session: Session, 
                 model_name: str = "mistral-large", 
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None):
        super().__init__()
        self.session = session
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def completion(self, 
                  model: str, 
                  messages: list, 
                  temperature: Optional[float] = None, 
                  max_tokens: Optional[int] = None, 
                  **kwargs) -> ModelResponse:
        """
        Execute completion using Snowflake's Cortex
        """
        try:
            # Extract the last message content as the prompt
            prompt = messages[-1]["content"]
            
            # Execute Snowflake query with just model and prompt
            # Note: Snowflake Cortex currently doesn't support temperature parameter
            response = self.session.sql(
                "SELECT snowflake.cortex.complete(?, ?)",
                params=(self.model_name, prompt)
            ).collect()[0][0]

            # Create response dictionary
            completion_response = {
                "id": f"snowflake-cortex-{int(time.time())}",
                "choices": [{
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": response,
                        "role": "assistant",
                        "function_call": None,
                        "tool_calls": None
                    }
                }],
                "created": int(time.time()),
                "model": model,
                "object": "chat.completion",
                "usage": {
                    "prompt_tokens": len(prompt.split()),  # Rough estimation
                    "completion_tokens": len(response.split()),  # Rough estimation
                    "total_tokens": len(prompt.split()) + len(response.split())  # Rough estimation
                }
            }
            
            return ModelResponse(**completion_response)
            
        except Exception as e:
            print(f"Error in Snowflake Cortex completion: {str(e)}")
            raise e

    async def acompletion(self, 
                         model: str, 
                         messages: list, 
                         temperature: Optional[float] = None, 
                         max_tokens: Optional[int] = None, 
                         **kwargs) -> ModelResponse:
        """
        Async version of completion - returns same response as sync version for now
        """
        return self.completion(model, messages, temperature, max_tokens, **kwargs)

class CrewSnowflakeLLM(LLM):
    """CrewAI compatible LLM implementation for Snowflake's Cortex"""
    
    def __init__(
        self, 
        session: Session,
        model_name: str = "mistral-large",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
    ):
        # Initialize base LLM with model identifier
        model = f"snowflake-cortex/{model_name}"
        super().__init__(model=model)
        
        # Initialize the custom LiteLLM implementation
        self.cortex_llm = SnowflakeCortexLLM(
            session=session,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Register the custom provider with LiteLLM
        litellm.custom_provider_map = [
            {
                "provider": "snowflake-cortex",
                "custom_handler": self.cortex_llm
            }
        ]
        
        # Store parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout

    def call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        callbacks: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Process the prompt and return a string response"""
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in LLM call: {str(e)}")
            raise e

    async def acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        callbacks: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Async version of call"""
        return self.call(prompt, stop, callbacks, **kwargs)

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM"""
        return "snowflake-cortex"

# Example usage
def test_snowflake_llm():
    try:
        # Create Snowflake session
        snowpark_session = Session.builder.configs(connection_params).create()
        
        # Initialize CrewAI compatible LLM
        llm = CrewSnowflakeLLM(
            session=snowpark_session,
            model_name="mistral-large",
            temperature=0.7
        )
        
        # Test the LLM
        response = llm.call("What is Mistral AI, and what models do they have?")
        print("Response:", response)
        
    except Exception as e:
        print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    test_snowflake_llm()
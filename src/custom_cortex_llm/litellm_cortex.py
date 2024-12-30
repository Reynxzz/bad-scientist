from snowflake.snowpark.session import Session
from dotenv import load_dotenv
import os
import litellm
from litellm import CustomLLM
from litellm.types.utils import ModelResponse
import time
from crewai import LLM
from typing import Optional, List, Union, Any
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
    
    def __init__(self, session: Session, model_name: str = "mistral-large", temperature: float = 0.7):
        super().__init__()
        self.session = session
        self.model_name = model_name
        self.temperature = temperature

    def completion(self, model: str, messages: list, temperature: float = None, max_tokens: int = None, **kwargs) -> ModelResponse:
        """
        Execute completion using Snowflake's Cortex
        """
        try:
            # Extract the last message content as the prompt
            prompt = messages[-1]["content"]
            
            # Execute Snowflake query
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
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            
            return ModelResponse(**completion_response)
            
        except Exception as e:
            print(f"Error in Snowflake Cortex completion: {str(e)}")
            raise e

    async def acompletion(self, model: str, messages: list, temperature: float = None, max_tokens: int = None, **kwargs) -> ModelResponse:
        """
        Async version of completion - returns same response as sync version for now
        """
        return self.completion(model, messages, temperature, max_tokens, **kwargs)

snowpark_session = Session.builder.configs(connection_params).create()

cortex_llm = SnowflakeCortexLLM(
    session=snowpark_session,
    model_name="mistral-large2",
    temperature=0.7
)

litellm.custom_provider_map = [
    {
        "provider": "snowflake-cortex",
        "custom_handler": cortex_llm
    }
]

# # Example usage
# def test_completion():
#     try:
#         response = litellm.completion(
#             model="snowflake-cortex/mistral-large2",
#             messages=[{"role": "user", "content": "What is Mistral AI, and what model they have?"}]
#         )
#         print("Response:", response.choices[0].message.content)
#     except Exception as e:
#         print(f"Error during completion: {str(e)}")

# if __name__ == "__main__":
#     test_completion()

from crewai import LLM
from typing import Optional, List, Union, Any, Dict
from langchain.callbacks.manager import CallbackManagerForLLMRun
import litellm

class CrewSnowflakeLLM(LLM):
    def __init__(
        self, 
        session, 
        model_name: str = "mistral-large2", 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
    ):
        # Initialize model identifier
        model = f"snowflake-cortex/{model_name}"
        super().__init__(model=model)
        
        # Initialize the custom LiteLLM implementation
        self.cortex_llm = SnowflakeCortexLLM(
            session=session,
            model_name=model_name,
            temperature=temperature
        )
        
        # Register the custom provider with LiteLLM
        litellm.custom_provider_map = [
            {
                "provider": "snowflake-cortex",
                "custom_handler": self.cortex_llm
            }
        ]
        
        # Set parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.supported_params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "request_timeout": request_timeout
        }

    def call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        callbacks: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Process the prompt and return a string."""
        messages = [{"role": "user", "content": prompt}]
        
        # Prepare parameters
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "request_timeout": self.request_timeout,
            **kwargs
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                **params
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
        """Async version of call."""
        return self.call(prompt, stop, callbacks, **kwargs)

    def get_model_name(self) -> str:
        """Get the model name"""
        return self.model

    def set_cache(self, cache: Optional[dict] = None) -> None:
        """Set cache for the LLM"""
        pass

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "snowflake-cortex"

    def get_supported_params(self) -> Dict:
        """Return supported parameters"""
        return self.supported_params
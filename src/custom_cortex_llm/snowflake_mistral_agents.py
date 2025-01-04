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

class SnowflakeCortexLLM(CustomLLM):
    """Custom LiteLLM implementation for Snowflake's Cortex"""
    
    def __init__(self, 
                 session: Session, 
                 model_name: str = "mistral-large2", 
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None):
        super().__init__()
        self.session = session
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Define supported parameters
        self.context_window = 4096  # Example value, adjust based on your model
        self.max_tokens_default = 2048  # Example value, adjust based on your model
        self._supported_params = {
            "temperature": {"type": float, "default": 0.7, "min": 0.0, "max": 1.0},
            "max_tokens": {"type": int, "default": self.max_tokens_default, "min": 1, "max": self.context_window},
        }

    def get_supported_params(self) -> Dict:
        """Return supported parameters"""
        return self._supported_params

    def _format_prompt(self, messages: Union[str, List[Dict[str, Any]]]) -> str:
        """Format messages into a single prompt string"""
        if isinstance(messages, str):
            return messages
            
        formatted_messages = []
        for message in messages:
            if isinstance(message.get('content'), list):
                # Handle nested message structure
                for submsg in message['content']:
                    if submsg['role'] == 'system':
                        formatted_messages.append(f"System: {submsg['content']}")
                    elif submsg['role'] == 'user':
                        formatted_messages.append(f"User: {submsg['content']}")
            else:
                # Handle simple message structure
                if message['role'] == 'system':
                    formatted_messages.append(f"System: {message['content']}")
                elif message['role'] == 'user':
                    formatted_messages.append(f"User: {message['content']}")
                elif message['role'] == 'assistant':
                    formatted_messages.append(f"Assistant: {message['content']}")
                
        return "\n".join(formatted_messages)

    def completion(self, 
                  model: str, 
                  messages: Union[str, List[Dict[str, Any]]], 
                  temperature: Optional[float] = None, 
                  max_tokens: Optional[int] = None, 
                  **kwargs) -> ModelResponse:
        """Execute completion using Snowflake's Cortex"""
        try:
            # Format the messages into a single prompt
            prompt = self._format_prompt(messages)
            
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
                        "role": "assistant"
                    }
                }],
                "created": int(time.time()),
                "model": model,
                "object": "chat.completion",
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(response.split()),
                    "total_tokens": len(prompt.split()) + len(response.split())
                }
            }
            
            return ModelResponse(**completion_response)
            
        except Exception as e:
            print(f"Error in Snowflake Cortex completion: {str(e)}")
            raise e

    async def acompletion(self, 
                         model: str, 
                         messages: Union[str, List[Dict[str, Any]]], 
                         temperature: Optional[float] = None, 
                         max_tokens: Optional[int] = None, 
                         **kwargs) -> ModelResponse:
        """Async version of completion"""
        return self.completion(model, messages, temperature, max_tokens, **kwargs)

class CrewSnowflakeLLM(LLM):
    """CrewAI compatible LLM implementation for Snowflake's Cortex"""
    
    def __init__(
        self, 
        session: Session,
        model_name: str = "mistral-large2",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
    ):
        model = f"snowflake-cortex/{model_name}"
        super().__init__(model=model)
        
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
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout

    def _format_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Format a prompt string into a messages list"""
        return [{"role": "user", "content": prompt}]

    def call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        callbacks: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Process the prompt and return a string response"""
        try:
            # Use the SnowflakeCortexLLM directly
            response = self.cortex_llm.completion(
                model=self.model,
                messages=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
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

    @property
    def supported_params(self) -> Dict:
        """Return supported parameters"""
        return self.cortex_llm.get_supported_params()
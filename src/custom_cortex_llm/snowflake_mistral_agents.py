import litellm
from litellm import CustomLLM, completion
from litellm.types.utils import ModelResponse
from snowflake.snowpark.session import Session
from crewai import LLM
from typing import Optional, List, Dict, Union
import time
from config import MODEL_NAME, MODEL_TEMPERATURE

class SnowflakeCortexLLM(CustomLLM):
    def __init__(self, session: Session, model_name: str = MODEL_NAME):
        super().__init__()
        self.session = session
        self.model_name = model_name

    def completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        try:
            prompt = self._format_messages(messages)
            
            # Call Snowflake Cortex
            response = self.session.sql(
                "SELECT snowflake.cortex.complete(?, ?)",
                params=(self.model_name, prompt)
            ).collect()[0][0]

            # Format the response
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
                "object": "chat.completion"
            }
            
            return ModelResponse(**completion_response)
        except Exception as e:
            print(f"Error in completion: {str(e)}")
            raise e

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        formatted = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if content:
                formatted.append(f"{role.capitalize()}: {content}")
        return "\n".join(formatted)
    
class CrewSnowflakeLLM(LLM):
    """
    Integration for using Snowflake's Cortex LLM with CrewAI agents via LiteLLM.
        - SnowflakeCortexLLM: Custom LLM implementation for Snowflake's Cortex
        - CrewSnowflakeLLM: CrewAI-compatible wrapper for SnowflakeCortexLLM

    Example:
        llm = CrewSnowflakeLLM(
            session=snowpark_session,
            model_name="your_model",
            temperature=0.7
        )
        agent = Agent(role="Assistant", llm=llm)
    """
    def __init__(
        self,
        session: Session,
        model_name: str = MODEL_NAME,
        temperature: float = MODEL_TEMPERATURE,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        self.cortex_llm = SnowflakeCortexLLM(session=session, model_name=model_name)
        
        litellm.custom_provider_map = [
            {
                "provider": "snowflake-cortex",
                "custom_handler": self.cortex_llm
            }
        ]
        
        model = f"snowflake-cortex/{model_name}"
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    def call(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        try:
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt

            # Use litellm's completion function
            response = completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in LLM call: {str(e)}")
            raise e
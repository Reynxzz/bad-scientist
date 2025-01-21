import litellm
from litellm import CustomLLM, completion
from litellm.types.utils import ModelResponse
from snowflake.snowpark.session import Session
from crewai import LLM
from typing import Optional, List, Dict, Union
import time
from config import MODEL_NAME, MODEL_TEMPERATURE
import json

class SnowflakeCortexLLM(CustomLLM):
    def __init__(self, session: Session, model_name: str = MODEL_NAME):
        super().__init__()
        self.session = session
        self.model_name = model_name
        self._supported_params = [
            "model",
            "messages",
            "temperature",
            "max_tokens",
            "top_p",
            "stream",
            "presence_penalty",
            "frequency_penalty"
        ]

    @property
    def supported_params(self):
        return self._supported_params
    
    def completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        try:
            prompt = self._format_messages(messages)
            
            messages = json.dumps([
                {
                    'role': 'system', 
                    'content': 'You are a helpful AI decision maker agent that understand user intention and know which tools or function (if required) you can use to fullfil user requiremnts'
                },
                {
                    'role': 'user', 
                    'content': prompt
                }
            ])
            
            parameters = json.dumps({                               
                'temperature': MODEL_TEMPERATURE,
            })
            
            result = self.session.sql(
                "SELECT snowflake.cortex.complete(?, parse_json(?), parse_json(?))",
                params=[self.model_name, messages, parameters]
            ).collect()[0][0]

            response = json.loads(result)
            
            content = ""
            if response and 'choices' in response and len(response['choices']) > 0:
                if isinstance(response['choices'][0], dict) and 'message' in response['choices'][0]:
                    content = response['choices'][0]['message'].get('content', '')
                elif isinstance(response['choices'][0], dict) and 'text' in response['choices'][0]:
                    content = response['choices'][0]['text']
                else:
                    content = str(response['choices'][0])

            completion_response = {
                "id": f"snowflake-cortex-{int(time.time())}",
                "choices": [{
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": content,
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
    def __init__(
        self,
        session,
        model_name: str,
        temperature: float = MODEL_TEMPERATURE,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        self.cortex_llm = SnowflakeCortexLLM(session=session, model_name=model_name)
        
        litellm.custom_provider_map = [
            {
                "provider": "snowflake-cortex",
                "custom_handler": self.cortex_llm,
                "supported_params": self.cortex_llm.supported_params
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

            params = {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            params.update(kwargs)

            response = completion(
                model=self.model,
                messages=messages,
                **params
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in LLM call: {str(e)}")
            raise e
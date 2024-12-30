from snowflake.snowpark.session import Session
from dotenv import load_dotenv
import os
import litellm
from litellm import CustomLLM
from litellm.types.utils import ModelResponse
import time

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
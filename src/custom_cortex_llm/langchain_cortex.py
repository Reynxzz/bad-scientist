from snowflake.snowpark.session import Session
from dotenv import load_dotenv
import os
# Initialize LLM to use Snowflake's Mistral model
from langchain.llms.base import BaseLLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema.output import Generation, LLMResult
from pydantic import BaseModel, Field, PrivateAttr
from snowflake.snowpark.session import Session

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

snowpark_session = Session.builder.configs(connection_params).create()

class MistralLLM(BaseLLM, BaseModel):
    """LangChain LLM implementation for Snowflake's Mistral model"""
    
    # Private attribute for session
    _session: Session = PrivateAttr()
    
    # Model configuration
    model_name: str = Field(default="mistral-large", description="Name of the Mistral model to use")
    temperature: float = Field(default=0.7, description="Temperature for text generation")
    
    def __init__(self, session: Session, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    def _call(self, prompt: str, **kwargs) -> str:
        """Execute single prompt completion"""
        try:
            response = self._session.sql(
                "SELECT snowflake.cortex.complete(?, ?)",
                params=(self.model_name, prompt)
            ).collect()[0][0]
            return response
        except Exception as e:
            print(f"Error calling Mistral: {str(e)}")
            raise e

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate completions for multiple prompts"""
        generations = []
        for prompt in prompts:
            response = self._call(prompt, **kwargs)
            generations.append([Generation(text=response)])
        return LLMResult(generations=generations)
            
    @property
    def _llm_type(self) -> str:
        return "mistral-large"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature
        }

# Create Mistral LLM instance
llm = MistralLLM(
    session=snowpark_session, 
    model_name="mistral-large2",
    temperature=0.7
)

# # USAGE
# res = llm._call('what is mistal AI')
# print(res)
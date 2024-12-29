from pydantic import BaseModel, Field
from typing import List, Optional

# --- Base Models ---
class StreamlitAppRequirements(BaseModel):
    app_name: str = Field(..., description="Name of the Streamlit application")
    components: List[str] = Field(..., description="Required Streamlit components")
    functionality: List[str] = Field(..., description="Core functionality requirements")
    data_requirements: Optional[List[str]] = Field(None, description="Data requirements")
    security_requirements: Optional[List[str]] = Field(None, description="Security requirements")

class GeneratedCode(BaseModel):
    file_name: str
    code_content: str
    dependencies: List[str]
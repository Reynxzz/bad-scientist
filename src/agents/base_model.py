from pydantic import BaseModel, Field
from typing import Optional, List

class RequirementAnalysis(BaseModel):
    components: List[str] = Field(..., description="List of identified technical components")
    dependencies: List[str] = Field(default_factory=list, description="Required dependencies")
    integration_points: List[str] = Field(default_factory=list, description="System integration requirements")

class TechnicalSpec(BaseModel):
    component_details: dict = Field(..., description="Detailed specifications for each component")
    architecture: str = Field(..., description="Proposed system architecture")
    tech_stack: List[str] = Field(..., description="Required technology stack")

class ValidationResult(BaseModel):
    passed: bool = Field(..., description="Whether the validation passed")
    issues: List[str] = Field(default_factory=list, description="List of identified issues")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")

class DocumentChunk(BaseModel):
    id: Optional[int] = None
    doc_text: str
    doc_type: str
    metadata: dict = Field(default_factory=dict)
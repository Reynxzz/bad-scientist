from snowflake.snowpark.session import Session
from snowflake.core import Root
from typing import List, Type, Optional
from pydantic import BaseModel, Field
from crewai_tools.tools.base_tool import BaseTool
from enum import Enum
from config import MODEL_NAME

class TechStack(str, Enum):
    STREAMLIT = "streamlit"
    ST_REF = "st_ref"

class DocumentType(Enum):
    REQUIREMENTS = "req_docs"
    STREAMLIT_DOCS = "streamlit_code"
    ST_REF_DOCS = "streamlit_appgalery"

class ReqSearchInput(BaseModel):
    """Input schema for document search."""
    query: str = Field(description="The search query to use")
    doc_type: str = Field(description="Type of document to search ('requirements')")

    model_config = {
        "json_schema_extra": {
            "examples": [{"query": "authentication flow", "doc_type": "requirements"}]
        }
    }

class SearchInput(BaseModel):
    """Input schema for document search."""
    query: str = Field(description="The search query to use")
    tech_stack: TechStack = Field(
        description="Technology stack to search ('streamlit' or 'st_ref')"
    )
    doc_type: str = Field(description="Type of document to search ('requirements' or 'technical_docs')")
    prev_context: Optional[str] = Field(description="Context from previous task, to learn or improve")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "authentication flow", 
                    "tech_stack": "streamlit",
                    "doc_type": "technical_docs",
                    "prev_context": "Initial OAuth implementation in streamlit... ```python  st.session_state ... ```"
                }
            ]
        }
    }

class SearchResult(BaseModel):
    """Schema for a single search result."""
    doc_text: str = Field(description="The text content of the document")
    source: str = Field(description="Source/origin of the document")

    model_config = {
        "json_schema_extra": {
            "examples": [{"doc_text": "Sample document text", "source": "requirements.txt"}]
        }
    }

class SearchOutput(BaseModel):
    """Output schema for document search with LLM processing."""
    context: str = Field(description="Combined search results context")
    response: str = Field(description="LLM-generated response based on the context")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "context": "Combined document context",
                "response": "LLM-generated answer"
            }]
        }
    }

class CortexSearchRequirementsTool(BaseTool):
    name: str = "Search Requirements Documents"
    description: str = "Search through business requirements and get LLM-processed answers"
    args_schema: Type[BaseModel] = ReqSearchInput
    return_schema: Type[BaseModel] = SearchOutput
    
    def __init__(self, snowpark_session: Session, result_as_answer: bool = False):
        super().__init__()
        self._session = snowpark_session
        self._root = Root(self._session)
        self.result_as_answer = result_as_answer

    def _run(self, query: str, doc_type: str = "requirements") -> SearchOutput:
        """Run the search and process results with LLM."""
        
        print(f"`CortexSearchRequirementsTool` called with query: {query}, doc_type: {doc_type}")

        service_name = f"{DocumentType.REQUIREMENTS.value}_search_svc"
        search_service = (
            self._root
            .databases[self._session.get_current_database()]
            .schemas[self._session.get_current_schema()]
            .cortex_search_services[service_name]
        )
        
        results = search_service.search(
            query=query,
            columns=["doc_text", "source"],
            limit=5
        )
        
        context = "\n\n".join([
            f"Source: {r['source']}\nContent: {r['doc_text']}"
            for r in results.results
        ])
        
        prompt = f"""
        Based on the following context, extract and analyze the key technical requirements (for MVP).
        Make it short and clear in less than 50 words. If possible, generate a template streamlit app to fulfil the requirement.

        Context:
        {context}

        Question: {query}

        Technical Requirements Analysis:
        - Identify all Python-implementable components
        - List required data sources and processing needs
        - Specify required Streamlit UI components
        - Note any performance or scalability requirements

        FORMAT YOUR RESPONSE AS:
        1. Technical Requirements: [List each requirement]
        2. Data Requirements: [List data needs]
        3. UI Components: [List Streamlit elements]
        4. Integration Needs: [List dependencies]
        5. Constraints: [List any limitations]
        """
        
        response = self._session.sql(
            "SELECT snowflake.cortex.complete(?, ?)",
            params=(MODEL_NAME, prompt)
        ).collect()[0][0]

        print("Requirement Tool Response:", response)

        return response

class CortexSearchTechnicalTool(BaseTool):
    name: str = "Search Technical Documentation"
    description: str = """Search through technical documentation and get implementation guidance."""
    args_schema: Type[BaseModel] = SearchInput
    return_schema: Type[BaseModel] = SearchOutput

    def __init__(self, snowpark_session: Session, result_as_answer: bool = False):
        super().__init__()
        self._session = snowpark_session
        self._root = Root(self._session)
        self.result_as_answer = result_as_answer

    def _run(self, query: str, tech_stack: str, doc_type: str = "technical_docs", prev_context: Optional[str] = None) -> SearchOutput:
        """Run the search and process results with LLM."""

        print(f"`CortexSearchTechnicalTool` called with query: {query}, doc_type: {doc_type}, tech_stack: {tech_stack}")

        if tech_stack == TechStack.STREAMLIT:
            service_name = f"{DocumentType.STREAMLIT_DOCS.value}_search_svc"
        elif tech_stack == TechStack.ST_REF:
            service_name = f"{DocumentType.ST_REF_DOCS.value}_search_svc"
        else:
            raise ValueError("tech_stack must be specified as either 'streamlit' or 'st_ref'")
            
        search_service = (
            self._root
            .databases[self._session.get_current_database()]
            .schemas[self._session.get_current_schema()]
            .cortex_search_services[service_name]
        )
        
        results = search_service.search(
            query=query,
            columns=["doc_text"],
            limit=5
        )
        
        context = "\n\n".join([
            f"Content: {r['doc_text']}"
            for r in results.results
        ])
        
        prompt = f"""
        Use the following {tech_stack} documentation as guidance, provide code implementation guidance.
        Documentation:
        {context}

        Question: {query}
        """
        
        response = self._session.sql(
            "SELECT snowflake.cortex.complete(?, ?)",
            params=(MODEL_NAME, prompt)
        ).collect()[0][0]

        print("Technical Tool Response:", response)

        return response

def create_search_tools(snowpark_session: Session) -> List[BaseTool]:
    """Create CrewAI tools for document search"""
    return [
        CortexSearchRequirementsTool(snowpark_session)(result_as_answer=True),
        CortexSearchTechnicalTool(snowpark_session)(result_as_answer=True)
    ]
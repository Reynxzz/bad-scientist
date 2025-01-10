from snowflake.snowpark.session import Session
from snowflake.core import Root
from typing import List, Type, Optional
from pydantic import BaseModel, Field, PrivateAttr
from crewai_tools.tools.base_tool import BaseTool
from enum import Enum
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")

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
    doc_type: str = Field(description="Type of document to search ('requirements' or 'technical_docs')")
    tech_stack: Optional[TechStack] = Field(
        None, 
        description="Technology stack to search ('streamlit' or 'st_ref')"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "authentication flow", 
                    "doc_type": "technical_docs",
                    "tech_stack": "streamlit"
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
        
        # Combine search results into context
        context = "\n\n".join([
            f"Source: {r['source']}\nContent: {r['doc_text']}"
            for r in results.results
        ])
        
        prompt = f"""
        Based on the following context, extract and analyze the key technical requirements.
        Make it short and clear in less than 50 words.

        Context:
        {context}

        Question: {query}
        """
        
        response = self._session.sql(
            "SELECT snowflake.cortex.complete(?, ?)",
            params=("mistral-large2", prompt)
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

    def _run(self, query: str, doc_type: str = "technical_docs", tech_stack: Optional[str] = None) -> SearchOutput:
        """Run the search and process results with LLM."""
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
        Based on the following {tech_stack} documentation, provide code implementation guidance.

        Documentation:
        {context}

        Question: {query}
        """
        
        response = self._session.sql(
            "SELECT snowflake.cortex.complete(?, ?)",
            params=("mistral-large2", prompt)
        ).collect()[0][0]

        print("Technical Tool Response:", response)

        return response

class DocumentProcessor:
    def __init__(self, snowpark_session: Session):
        self.session = snowpark_session
        self.root = Root(self.session)
        self.init_snowflake_storage()
    
    def init_snowflake_storage(self):
        """Initialize Snowflake tables and search services"""
        self.session.sql(f"USE DATABASE {DATABASE}").collect()
        self.session.sql(f"USE SCHEMA {SCHEMA}").collect()
        
        self._create_document_table(DocumentType.REQUIREMENTS)
        self._create_search_service(DocumentType.REQUIREMENTS)
        print("Storing PDF...")

    def _create_document_table(self, doc_type: DocumentType):
        self.session.sql(f"""
            CREATE TABLE IF NOT EXISTS {doc_type.value}_chunks (
                id INTEGER AUTOINCREMENT,
                doc_text STRING,
                source STRING,
                metadata VARIANT
            )
        """).collect()
        print("Chunked PDF Table Created...")

    def _create_search_service(self, doc_type: DocumentType):
        self.session.sql(f"""
            CREATE OR REPLACE CORTEX SEARCH SERVICE {doc_type.value}_search_svc
            ON doc_text
            WAREHOUSE = {WAREHOUSE}
            TARGET_LAG = '1 hour'
            AS 
                SELECT 
                    doc_text,
                    source,
                    metadata
                FROM {doc_type.value}_chunks
        """).collect()
        print("Chunked PDF Search Service Created...")

    def process_document(self, file_path: str, doc_type: DocumentType, source: str = None) -> List[str]:
        """Process document and split into chunks"""
        if file_path.lower().endswith('.pdf'):
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            content = '\n\n'.join([page.page_content for page in pages])
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
        chunks = self._split_into_chunks(content)
        self._store_chunks(chunks, doc_type, source or file_path)
        return chunks

    def _split_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into manageable chunks using LangChain's RecursiveCharacterTextSplitter
        
        Args:
            text: The input text to be split
            chunk_size: Maximum size of each chunk in characters
            
        Returns:
            List of text chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        print("PDF splitted into chunks...")
        return chunks

    def _store_chunks(self, chunks: List[str], doc_type: DocumentType, source: str):
        """Store document chunks in Snowflake"""
        table_name = f"{doc_type.value}_chunks"
        
        # Truncate the existing table
        self.session.sql(f"TRUNCATE TABLE IF EXISTS {table_name}").collect()
        print(f"Cleared existing data from {table_name}")
        
        # Batch insert chunks
        values_list = [(chunk, source, '{}') for chunk in chunks]
        placeholders = ", ".join(["(?, ?, ?)"] * len(values_list))
        flattened_params = [param for value_tuple in values_list for param in value_tuple]
        
        self.session.sql(f"""
            INSERT INTO {table_name} (doc_text, source, metadata)
            SELECT column1, column2, PARSE_JSON(column3)
            FROM VALUES {placeholders}
        """, params=flattened_params).collect()
        
        print(f"Stored {len(chunks)} new chunks in {table_name}")

def create_search_tools(snowpark_session: Session) -> List[BaseTool]:
    """Create CrewAI tools for document search"""
    return [
        CortexSearchRequirementsTool(snowpark_session)(result_as_answer=True),
        CortexSearchTechnicalTool(snowpark_session)(result_as_answer=True)
    ]
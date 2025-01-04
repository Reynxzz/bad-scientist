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
    SKLEARN = "sklearn"

class DocumentType(Enum):
    REQUIREMENTS = "req_docs"
    STREAMLIT_DOCS = "streamlit_docs"
    SKLEARN_DOCS = "sklearn_docs"

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
        description="Technology stack to search ('streamlit' or 'sklearn')"
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
    """Output schema for document search."""
    results: List[SearchResult] = Field(description="List of search results")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "results": [
                    {"doc_text": "Sample document text", "source": "requirements.txt"}
                ]
            }]
        }
    }

class CortexSearchRequirementsTool(BaseTool):
    """Tool for searching through business requirements documents."""
    name: str = "Search Requirements Documents"
    description: str = "Search through business requirements and project documentation"
    args_schema: Type[BaseModel] = ReqSearchInput
    return_schema: Type[BaseModel] = SearchOutput
    
    # Use private attributes for session and root
    _session: Session = PrivateAttr()
    _root: Root = PrivateAttr()
    
    def __init__(self, snowpark_session: Session):
        super().__init__()
        self._session = snowpark_session
        self._root = Root(self._session)

    def _run(self, query: str, doc_type: str = "requirements") -> SearchOutput:
        """Run the search on requirements documents."""
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
        
        search_results = [
            SearchResult(doc_text=r['doc_text'], source=r['source'])
            for r in results.results
        ]
        
        return SearchOutput(results=search_results)

class CortexSearchTechnicalTool(BaseTool):
    """Tool for searching through technical documentation."""
    name: str = "Search Technical Documentation"
    description: str = """Search through technical documentation and implementation guides.
    Specify tech_stack as 'streamlit' or 'sklearn' to search the corresponding documentation."""
    args_schema: Type[BaseModel] = SearchInput
    return_schema: Type[BaseModel] = SearchOutput

    _session: Session = PrivateAttr()
    _root: Root = PrivateAttr()

    def __init__(self, snowpark_session: Session):
        super().__init__()
        self._session = snowpark_session
        self._root = Root(self._session)

    def _run(self, query: str, doc_type: str = "technical_docs", tech_stack: Optional[str] = None) -> SearchOutput:
        """Run the search on technical documentation."""
        # Determine which service to use based on tech_stack
        if tech_stack == TechStack.STREAMLIT:
            service_name = f"{DocumentType.STREAMLIT_DOCS.value}_search_svc"
        elif tech_stack == TechStack.SKLEARN:
            service_name = f"{DocumentType.SKLEARN_DOCS.value}_search_svc"
        else:
            raise ValueError("tech_stack must be specified as either 'streamlit' or 'sklearn'")
            
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
        
        search_results = [
            SearchResult(
                doc_text=r['doc_text'],
                source=tech_stack  # Use tech_stack as source for tracking
            )
            for r in results.results
        ]
        
        return SearchOutput(results=search_results)

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
        CortexSearchRequirementsTool(snowpark_session),
        CortexSearchTechnicalTool(snowpark_session)
    ]
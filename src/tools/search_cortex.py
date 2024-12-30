from snowflake.snowpark.session import Session
from snowflake.core import Root
from typing import List, Type, Optional
from pydantic import BaseModel, Field, PrivateAttr
from crewai_tools.tools.base_tool import BaseTool
from dotenv import load_dotenv
import enum
import os

load_dotenv()

DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")

class DocumentType(enum.Enum):
    REQUIREMENTS = "req_docs"
    TECHNICAL_DOCS = "streamlit_docs"

class SearchInput(BaseModel):
    """Input schema for document search."""
    query: str = Field(description="The search query to use")
    doc_type: str = Field(description="Type of document to search ('requirements' or 'technical_docs')")

    model_config = {
        "json_schema_extra": {
            "examples": [{"query": "authentication flow", "doc_type": "requirements"}]
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
    args_schema: Type[BaseModel] = SearchInput
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
    description: str = "Search through technical documentation and implementation guides"
    args_schema: Type[BaseModel] = SearchInput
    return_schema: Type[BaseModel] = SearchOutput

    # Use private attributes for session and root
    _session: Session = PrivateAttr()
    _root: Root = PrivateAttr()

    def __init__(self, snowpark_session: Session):
        super().__init__()
        self._session = snowpark_session
        self._root = Root(self._session)

    def _run(self, query: str, doc_type: str = "technical_docs") -> SearchOutput:
        """Run the search on technical documentation."""
        service_name = f"{DocumentType.TECHNICAL_DOCS.value}_search_svc"
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
        self._create_document_table(DocumentType.TECHNICAL_DOCS)
        
        self._create_search_service(DocumentType.REQUIREMENTS)
        self._create_search_service(DocumentType.TECHNICAL_DOCS)

    def _create_document_table(self, doc_type: DocumentType):
        if doc_type.value == 'req_docs':
            self.session.sql(f"""
                CREATE TABLE IF NOT EXISTS {doc_type.value}_chunks (
                    id INTEGER AUTOINCREMENT,
                    doc_text STRING,
                    source STRING,
                    metadata VARIANT
                )
            """).collect()
        else:
            pass

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
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word)
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def _store_chunks(self, chunks: List[str], doc_type: DocumentType, source: str):
        """Store document chunks in Snowflake"""
        table_name = f"{doc_type.value}_chunks"
        for chunk in chunks:
            self.session.sql(f"""
                INSERT INTO {table_name} (doc_text, source, metadata)
                SELECT column1, column2, PARSE_JSON(column3)
                FROM VALUES(?, ?, ?)
            """, params=(chunk, source, '{}')).collect()

def create_search_tools(snowpark_session: Session) -> List[BaseTool]:
    """Create CrewAI tools for document search"""
    return [
        CortexSearchRequirementsTool(snowpark_session),
        CortexSearchTechnicalTool(snowpark_session)
    ]
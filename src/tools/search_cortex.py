from snowflake.snowpark.session import Session
from snowflake.core import Root
from langchain.tools import Tool
from typing import List
import enum
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")

class DocumentType(enum.Enum):
    REQUIREMENTS = "doc_chunks"
    TECHNICAL_DOCS = "streamlit_docs"

class CortexSearchTool:
    def __init__(self, snowpark_session: Session, doc_type: DocumentType):
        self.session = snowpark_session
        self.root = Root(self.session)
        self.doc_type = doc_type
        
    def search(self, query: str) -> str:
        """Search documents using Cortex Search"""
        service_name = f"{self.doc_type.value}_search_svc"
        search_service = (
            self.root
            .databases[self.session.get_current_database()]
            .schemas[self.session.get_current_schema()]
            .cortex_search_services[service_name]
        )
        
        results = search_service.search(
            query=query,
            columns=["doc_text", "source"],
            limit=5
        )
        
        # Format results into a readable string
        formatted_results = "\n\n".join([
            f"Document ({r['source']}): {r['doc_text']}" 
            for r in results.results
        ])
        return formatted_results

class DocumentProcessor:
    def __init__(self, snowpark_session: Session):
        self.session = snowpark_session
        self.root = Root(self.session)
        self.init_snowflake_storage()
    
    def init_snowflake_storage(self):
        """Initialize Snowflake tables and search services"""
        self.session.sql(f"USE DATABASE {DATABASE}").collect()
        self.session.sql(f"USE SCHEMA {SCHEMA}").collect()
        
        # Create tables for different document types
        self._create_document_table(DocumentType.REQUIREMENTS)
        self._create_document_table(DocumentType.TECHNICAL_DOCS)
        
        # Create search services for each document type
        self._create_search_service(DocumentType.REQUIREMENTS)
        self._create_search_service(DocumentType.TECHNICAL_DOCS)

    def _create_document_table(self, doc_type: DocumentType):
        """Create table for specific document type"""
        self.session.sql(f"""
            CREATE TABLE IF NOT EXISTS {doc_type.value}_chunks (
                id INTEGER AUTOINCREMENT,
                doc_text STRING,
                source STRING,
                metadata VARIANT
            )
        """).collect()

    def _create_search_service(self, doc_type: DocumentType):
        """Create Cortex Search service for specific document type"""
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

    def ingest_technical_docs(self, docs_dir: str):
        """Ingest technical documentation from a directory"""
        from pathlib import Path
        
        # Walk through the docs directory
        for file_path in Path(docs_dir).rglob('*'):
            if file_path.is_file() and file_path.suffix in ['.md', '.txt', '.pdf']:
                try:
                    self.process_document(
                        str(file_path),
                        DocumentType.TECHNICAL_DOCS,
                        source=str(file_path.relative_to(docs_dir))
                    )
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

def create_search_tools(snowpark_session: Session) -> List[Tool]:
    """Create LangChain tools for document search"""
    requirements_tool = CortexSearchTool(snowpark_session, DocumentType.REQUIREMENTS)
    technical_tool = CortexSearchTool(snowpark_session, DocumentType.TECHNICAL_DOCS)
    
    return [
        Tool(
            name="search_requirements",
            func=requirements_tool.search,
            description="Search through business requirements documents"
        ),
        Tool(
            name="search_technical_docs",
            func=technical_tool.search,
            description="Search through technical documentation and implementation guides"
        )
    ]
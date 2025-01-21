from snowflake.snowpark.session import Session
from snowflake.core import Root
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import DATABASE, SCHEMA, WAREHOUSE
from enum import Enum
from typing import List

class DocumentType(Enum):
    REQUIREMENTS = "req_docs"
    STREAMLIT_DOCS = "streamlit_code"
    ST_REF_DOCS = "streamlit_appgalery"

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
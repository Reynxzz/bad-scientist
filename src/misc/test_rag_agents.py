import os
from typing import List
from crewai import Agent, Task, Crew
from langchain_community.document_loaders import PyPDFLoader
from snowflake.snowpark.session import Session
from snowflake.core import Root
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain_community.document_loaders import PyPDFLoader
from snowflake.snowpark.session import Session
from snowflake.core import Root
import tempfile

load_dotenv()

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


# -- Agentic RAG Class -- #

class PDFProcessingAgent:
    def __init__(self, snowpark_session: Session):
        self.session = snowpark_session
        self.root = Root(self.session)
        
        # Initialize Snowflake table for storing PDF chunks
        self.init_snowflake_storage()
        
    def init_snowflake_storage(self):
        """Initialize Snowflake table for storing PDF text chunks"""
        # Set the database context
        self.session.sql("USE DATABASE BADSCIENTIST_DEMO").collect()
        self.session.sql("USE SCHEMA NOTEBOOKS").collect()
        
        # Create table for storing PDF chunks
        self.session.sql("""
            CREATE TABLE IF NOT EXISTS pdf_docs (
                id INTEGER AUTOINCREMENT,
                doc_text STRING,
                metadata VARIANT
            )
        """).collect()
        
        # Create or replace Cortex Search service
        self.session.sql("""
            CREATE OR REPLACE CORTEX SEARCH SERVICE BADSCIENTIST_DEMO.NOTEBOOKS.pdf_docs_svc
            ON doc_text
            WAREHOUSE = compute_wh
            TARGET_LAG = '1 hour'
            AS 
                SELECT 
                    doc_text
                FROM pdf_docs
        """).collect()

    def process_pdf(self, pdf_path: str) -> List[str]:
        """Process PDF file and split into chunks"""
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # Split pages into smaller chunks for better retrieval
        chunks = []
        for page in pages:
            # Split into paragraphs or smaller segments
            text_chunks = self._split_into_chunks(page.page_content)
            chunks.extend(text_chunks)
            
        return chunks
    
    def _split_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into smaller chunks"""
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

    def store_chunks(self, chunks: List[str]):
        """Store text chunks in Snowflake"""
        for chunk in chunks:
            # Insert chunk into Snowflake
            self.session.sql("""
                INSERT INTO pdf_docs (doc_text, metadata)
                SELECT column1, PARSE_JSON(column2)
                FROM VALUES(?, ?)
            """, params=(chunk, '{}')).collect()

class PDFQueryAgent:
    def __init__(self, snowpark_session: Session):
        self.session = snowpark_session
        self.root = Root(self.session)
        
    def query(self, question: str, num_chunks: int = 4) -> dict:
        """Query the stored PDF content using Cortex Search"""
        # Get relevant chunks using Cortex Search
        search_service = (
            self.root
            .databases[self.session.get_current_database()]
            .schemas[self.session.get_current_schema()]
            .cortex_search_services['pdf_docs_svc']
        )
        
        results = search_service.search(
            query=question,
            columns=["doc_text"],
            limit=num_chunks
        )
        
        context = "\n\n".join([r["doc_text"] for r in results.results])
        
        # Generate response using Snowflake's LLM
        prompt = f"""
        Based on the following context, answer the question.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        print(prompt)
        response = self.session.sql(
            "SELECT snowflake.cortex.complete(?, ?)",
            params=("mistral-large", prompt)
        ).collect()[0][0]
        
        return {
            'question': question,
            'context': context,
            'response': response
        }

def create_pdf_tools(snowpark_session: Session):
    """Create LangChain tools for PDF processing and querying"""
    pdf_processor = PDFProcessingAgent(snowpark_session)
    pdf_querier = PDFQueryAgent(snowpark_session)
    
    process_tool = Tool(
        name="process_pdf",
        func=lambda pdf_path: pdf_processor.process_pdf(pdf_path),
        description="Process a PDF file and split it into chunks"
    )
    
    store_tool = Tool(
        name="store_chunks",
        func=lambda chunks: pdf_processor.store_chunks(chunks),
        description="Store text chunks in Snowflake database"
    )
    
    query_tool = Tool(
        name="query_pdf",
        func=lambda question: pdf_querier.query(question),
        description="Query PDF content and get answers"
    )
    
    return [process_tool, store_tool, query_tool]

# Create CREW AI agents
def create_pdf_crew(snowpark_session: Session):
    tools = create_pdf_tools(snowpark_session)
    
    # PDF Processing Agent
    pdf_processor = Agent(
        name="PDF Processor",
        role="Processes and stores PDF documents in Snowflake",
        goal="Efficiently process PDFs and store them for later retrieval",
        backstory="Expert at handling PDFs and managing document storage",
        tools=tools[:2]  # process_pdf and store_chunks tools
    )
    
    # Query Agent
    query_agent = Agent(
        name="Query Expert",
        role="Answers questions about stored PDF content",
        goal="Provide accurate answers based on stored PDF content",
        backstory="Specialist in information retrieval and question answering",
        tools=[tools[2]]  # query_pdf tool
    )
    
    # Create crew
    crew = Crew(
        agents=[pdf_processor, query_agent],
        tasks=[
            Task(
                description="Process and store PDF documents",
                agent=pdf_processor,
                expected_output="A confirmation message that the PDF has been processed and stored in the database"
            ),
            Task(
                description="Answer questions about PDF content",
                agent=query_agent,
                expected_output="A detailed answer to the user's question based on the content stored in the database"
            )
        ]
    )
    
    return crew


def handle_pdf_upload(pdf_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        
        processor = PDFProcessingAgent(snowpark_session)
        chunks = processor.process_pdf(tmp_path)
        processor.store_chunks(chunks)
        os.unlink(tmp_path)
        
        print(f"Successfully processed PDF and stored {len(chunks)} chunks")
        return "PDF processed and stored successfully!"
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise

# Query the content
def ask_question(question: str):
    query_agent = PDFQueryAgent(snowpark_session)
    result = query_agent.query(question)
    return result['response']

# Initialize
snowpark_session = Session.builder.configs(connection_params).create()
crew = create_pdf_crew(snowpark_session)

# Process a PDF
with open('src/RAG-paper.pdf', 'rb') as f:
    result = handle_pdf_upload(f)
print(result)

# Ask questions
response = ask_question("What is retrieval, how can I use python to code this concept")
print(response)
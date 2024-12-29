import os
from typing import List, Optional
from pathlib import Path
import streamlit as st
from crewai import Agent, Task, Crew, Process
from snowflake.snowpark.session import Session
from snowflake.core import Root
from dotenv import load_dotenv
import tempfile
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.tools import Tool

# Load environment variables
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

# Initialize Snowflake session
snowpark_session = Session.builder.configs(connection_params).create()

class DocumentProcessor:
    def __init__(self, snowpark_session: Session):
        self.session = snowpark_session
        self.root = Root(self.session)
        self.init_snowflake_storage()
    
    def init_snowflake_storage(self):
        """Initialize Snowflake tables and search service"""
        self.session.sql("USE DATABASE BADSCIENTIST_DEMO").collect()
        self.session.sql("USE SCHEMA NOTEBOOKS").collect()
        
        # Create table for storing document chunks
        self.session.sql("""
            CREATE TABLE IF NOT EXISTS doc_chunks (
                id INTEGER AUTOINCREMENT,
                doc_text STRING,
                doc_type STRING,
                metadata VARIANT
            )
        """).collect()
        
        # Create Cortex Search service
        self.session.sql("""
            CREATE OR REPLACE CORTEX SEARCH SERVICE doc_search_svc
            ON doc_text
            WAREHOUSE = compute_wh
            TARGET_LAG = '1 hour'
            AS 
                SELECT 
                    doc_text,
                    doc_type,
                    metadata
                FROM doc_chunks
        """).collect()

    def process_document(self, file_path: str, doc_type: str) -> List[str]:
        """Process document and split into chunks"""
        if file_path.lower().endswith('.pdf'):
            # Handle PDF files
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            content = '\n\n'.join([page.page_content for page in pages])
        else:
            # Handle text files
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
        chunks = self._split_into_chunks(content)
        self._store_chunks(chunks, doc_type)
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
    
    def _store_chunks(self, chunks: List[str], doc_type: str):
        """Store document chunks in Snowflake"""
        for chunk in chunks:
            self.session.sql("""
                INSERT INTO doc_chunks (doc_text, doc_type, metadata)
                SELECT column1, column2, PARSE_JSON(column3)
                FROM VALUES(?, ?, ?)
            """, params=(chunk, doc_type, '{}')).collect()

class CortexSearchTool:
    def __init__(self, snowpark_session: Session):
        self.session = snowpark_session
        self.root = Root(self.session)
        
    def search(self, query: str) -> str:
        """Search documents using Cortex Search"""
        search_service = (
            self.root
            .databases[self.session.get_current_database()]
            .schemas[self.session.get_current_schema()]
            .cortex_search_services['doc_search_svc']
        )
        
        results = search_service.search(
            query=query,
            columns=["doc_text", "doc_type"],
            limit=5
        )
        
        # Format results into a readable string
        formatted_results = "\n\n".join([f"Document ({r['doc_type']}): {r['doc_text']}" 
                                       for r in results.results])
        return formatted_results

class RequirementAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            role="Requirement Analyzer",
            goal="Analyze business requirements and extract key technical components",
            backstory="Expert at analyzing business requirements and breaking them down into technical components",
            llm=llm,
            verbose=True
        )

class ResearcherAgent(Agent):
    def __init__(self, llm, snowpark_session):
        # Create search tool
        search_tool = CortexSearchTool(snowpark_session)
        tools = [
            Tool(
                name="search_documents",
                func=search_tool.search,
                description="Search through documents using Cortex Search"
            )
        ]
        
        super().__init__(
            role="Technical Researcher",
            goal="Research and provide implementation details for technical components",
            backstory="Specialized in researching technical solutions and best practices",
            llm=llm,
            tools=tools,
            verbose=True
        )

class CoderAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            role="Code Generator",
            goal="Generate high-quality code based on technical specifications",
            backstory="Expert programmer specialized in implementing technical solutions",
            llm=llm,
            verbose=True
        )

class ValidatorAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            role="Code Validator",
            goal="Test and validate generated code",
            backstory="Quality assurance specialist focused on code testing and validation",
            llm=llm,
            verbose=True
        )

class FallbackAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            role="Error Handler",
            goal="Handle errors and provide alternative solutions",
            backstory="Specialized in debugging and providing fallback solutions",
            llm=llm,
            verbose=True
        )

def create_crew(prompt: str, docs_path: Optional[str] = None):
    """Create and configure the agent crew"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    # Initialize agents
    requirement_agent = RequirementAgent(llm)
    researcher_agent = ResearcherAgent(llm, snowpark_session)
    coder_agent = CoderAgent(llm)
    validator_agent = ValidatorAgent(llm)
    fallback_agent = FallbackAgent(llm)
    
    # Process documents if provided
    if docs_path:
        doc_processor = DocumentProcessor(snowpark_session)
        doc_processor.process_document(docs_path, "requirements")
    
    # Define tasks
    tasks = [
        Task(
            description=f"Analyze the following business requirement and extract key technical components: {prompt}",
            expected_output="A detailed list of technical components and requirements extracted from the business requirements",
            agent=requirement_agent
        ),
        Task(
            description="Research implementation details for the identified technical components using the search_documents tool",
            expected_output="Comprehensive technical specifications and implementation details for each component",
            agent=researcher_agent
        ),
        Task(
            description="Generate code based on the technical specifications",
            expected_output="Complete, working code implementation that meets the technical specifications",
            agent=coder_agent
        ),
        Task(
            description="Test and validate the generated code",
            expected_output="Test results and validation report for the implemented code",
            agent=validator_agent
        ),
        Task(
            description="Handle any remaining errors or issues",
            expected_output="Resolution for any identified issues and final optimization suggestions",
            agent=fallback_agent
        )
    ]
    
    # Create crew
    crew = Crew(
        agents=[requirement_agent, researcher_agent, coder_agent, validator_agent, fallback_agent],
        tasks=tasks,
        process=Process.sequential
    )
    
    return crew

# Streamlit UI
def main():
    st.title("Multi-Agent RAG Application Builder")
    
    # Business idea prompt input
    prompt = st.text_area("Enter your business idea prompt:")
    
    # Optional document upload
    uploaded_file = st.file_uploader("Upload business requirements document (optional)", type=['pdf', 'txt'])
    
    if st.button("Generate App"):
        if prompt:
            # Handle document upload
            docs_path = None
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    docs_path = tmp_file.name
            
            try:
                # Create and execute crew
                crew = create_crew(prompt, docs_path)
                result = crew.kickoff()
                
                # Display results
                st.write("Generated Application:")
                st.code(result)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                
            finally:
                # Cleanup
                if docs_path:
                    os.unlink(docs_path)
        else:
            st.error("Please enter a business idea prompt")

if __name__ == "__main__":
    main()
from typing import Optional
from pathlib import Path
import streamlit as st
from crewai import Crew, Task, Process
from langchain_openai import ChatOpenAI
import tempfile
import os

# Import agents
from agents.requirements.requirements import RequirementAgent
from agents.researcher.researcher import ResearcherAgent
from agents.coder.coder import CoderAgent
# from agents.validator.validator import ValidatorAgent

# Import tools
from tools.search_cortex import create_search_tools, DocumentProcessor, DocumentType

def create_crew(prompt: str, docs_path: Optional[str] = None):
    """Create and configure the agent crew"""
    # Initialize Snowflake session
    from snowflake.snowpark.session import Session
    from dotenv import load_dotenv
    
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
    
    snowpark_session = Session.builder.configs(connection_params).create()
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Create search tools
    tools = create_search_tools(snowpark_session)
    
    # Initialize agents
    requirement_agent = RequirementAgent(llm)
    researcher_agent = ResearcherAgent(llm, tools)
    coder_agent = CoderAgent(llm)
    # validator_agent = ValidatorAgent(llm)
    
    # Process documents if provided
    if docs_path:
        doc_processor = DocumentProcessor(snowpark_session)
        # Use the DocumentType enum instead of string
        doc_processor.process_document(docs_path, DocumentType.REQUIREMENTS)
    
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
        # Task(
        #     description="Test and validate the generated code",
        #     expected_output="Test results and validation report for the implemented code",
        #     agent=validator_agent
        # )
    ]
    
    # Create crew
    crew = Crew(
        agents=[requirement_agent, researcher_agent, coder_agent], #validator_agent
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )
    
    return crew

def main():
    st.title("Multi-Agent RAG Application Builder")
    
    # Business idea prompt input
    prompt = st.text_area("Enter your business idea prompt:", 
                         help="Describe your business idea or requirements in detail")
    
    # Optional document upload
    uploaded_file = st.file_uploader(
        "Upload business requirements document (optional)",
        type=['pdf', 'txt'],
        help="Upload additional documentation to provide more context"
    )
    
    # Project name input
    project_name = st.text_input(
        "Project Name",
        help="Enter a name for your project"
    )
    
    if st.button("Generate App", type="primary"):
        if prompt and project_name:
            try:
                with st.spinner("Processing your request..."):
                    # Handle document upload
                    docs_path = None
                    if uploaded_file:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            docs_path = tmp_file.name
                    
                    # Create and execute crew
                    crew = create_crew(prompt, docs_path)
                    
                    # Execute tasks
                    result = crew.kickoff()
                    
                    # Display results
                    st.header("Generated Application")
                    st.markdown(result)
                    
                    # Save results option
                    if st.button("Save Results"):
                        # Create directory for project
                        project_dir = Path(f"projects/{project_name}")
                        project_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save complete result
                        with open(project_dir / "output.md", "w") as f:
                            f.write(str(result))
                        
                        st.success(f"Results saved to {project_dir}/output.md")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
            finally:
                # Cleanup
                if docs_path:
                    try:
                        os.unlink(docs_path)
                    except Exception as e:
                        st.warning(f"Could not remove temporary file: {str(e)}")
        else:
            if not prompt:
                st.error("Please enter a business idea prompt")
            if not project_name:
                st.error("Please enter a project name")

if __name__ == "__main__":
    main()
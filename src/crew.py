from typing import Optional
from pathlib import Path
import streamlit as st
from crewai import Crew, Task, Process, LLM
from langchain_openai import ChatOpenAI
# from langchain_mistralai import ChatMistralAI
import tempfile
import os

# Import agents
from agents.requirements.requirements import RequirementAgent
from agents.researcher.researcher import ResearcherAgent
from agents.coder.coder import CoderAgent
# from agents.validator.validator import ValidatorAgent
# from langchain_google_genai import ChatGoogleGenerativeAI
from custom_cortex_llm.litellm_cortex import CrewSnowflakeLLM

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

    # llm = LLM(
    #     model="mistral/mistral-large-latest",
    # ),

    # llm = CrewSnowflakeLLM(
    #     session=snowpark_session,
    #     model_name="mistral-large2",
    #     temperature=0.7,
    #     max_tokens=2048,
    #     request_timeout=30
    # )
    llm = LLM(
        model="gemini/gemini-1.5-flash",
        temperature=0.7
    )
    
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Create search tools
    tools = create_search_tools(snowpark_session)
    
    # Initialize agents
    requirement_agent = RequirementAgent(llm, tools)
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
            description="""Analyze the business requirements using the Search Requirements Documents tool.
            Input: {prompt}
            
            Steps:
            1. Use the Search Requirements Documents tool to query relevant requirements with:
            - query: Extract key phrases from the prompt
            - doc_type: "requirements"
            2. Analyze and extract key technical components from the search results
            3. Compile findings into a structured list""",
            expected_output="A detailed list of technical components and requirements extracted from the business requirements",
            agent=requirement_agent,
            tools=[create_search_tools(snowpark_session)[0]]  # Requirements search tool
        ),
        Task(
            description="""Research technical implementation details using the Search Technical Documentation tool.
            
            Steps:
            1. For each technical component identified:
            - Use the Search Technical Documentation tool with:
                - query: Component name and key technical terms
                - doc_type: "technical_docs"
            2. Analyze the search results to extract implementation patterns and specifications
            3. Compile findings into comprehensive technical documentation""",
            expected_output="Comprehensive technical specifications and implementation details for each component",
            agent=researcher_agent,
            tools=[create_search_tools(snowpark_session)[1]]  # Technical docs search tool
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
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
    requirement_task = Task(
        description=f"""Analyze the business requirements using the Search Requirements Documents tool.
        Input: {prompt}
        
        Steps:
        1. Use the Search Requirements Documents tool to query relevant requirements with:
        - query: Extract key phrases from the prompt
        - doc_type: "requirements"
        2. Analyze and extract key technical components to implement using Python only from the search results
        3. Compile findings into a structured list with clear implementation requirements
        4. Each requirement must be tagged with its priority and technical scope""",
        expected_output="""
        1. Detailed list of technical components and requirements to implement using Python only
        2. Clear mapping of business requirements to technical components
        3. Prioritized implementation roadmap""",
        agent=requirement_agent,
        tools=[create_search_tools(snowpark_session)[0]]
        )
    
    researcher_sklearn_task = Task(
        description="""Research scikit-learn implementation details based on PREVIOUSLY IDENTIFIED REQUIREMENTS using the Search Technical Documentation tool.
        
        Context: Use the output from the requirements analysis task as your foundation.
        
        Steps:
        1. For each technical component previously identified in the requirements:
        - Use the Search Technical Documentation tool with:
            - query: Component name and key technical terms from requirements
            - doc_type: "technical_docs"
            - tech_stack: "sklearn"
        2. Map each scikit-learn implementation pattern directly to the original requirements
        3. Document how each implementation choice fulfills specific requirements""",
        expected_output="""
        1. Scikit-learn technical specifications mapped to original requirements
        2. Implementation patterns with direct traceability to business needs
        3. Technical documentation that maintains alignment with requirements""",
        agent=researcher_agent,
        context=[requirement_task],
        tools=[create_search_tools(snowpark_session)[1]]
        )
    
    researcher_streamlit_task = Task(
        description="""Research Streamlit implementation details based on BOTH REQUIREMENTS AND SKLEARN SPECIFICATIONS using the Search Technical Documentation tool.
        
        Context: Use outputs from both previous requirements and scikit-learn research tasks.
        
        Steps:
        1. For each UI component needed to fulfill the original requirements:
        - Use the Search Technical Documentation tool with:
            - query: Component requirements and sklearn integration points
            - doc_type: "technical_docs"
            - tech_stack: "streamlit"
        2. Ensure UI patterns align with both business requirements and sklearn implementation
        3. Document the connection between UI components and underlying sklearn functionality""",
        expected_output="""
        1. Streamlit technical specifications that align with original requirements
        2. UI implementation patterns that support sklearn integration
        3. Comprehensive documentation linking UI, sklearn, and business requirements""",
        agent=researcher_agent,
        context=[requirement_task, researcher_sklearn_task],
        tools=[create_search_tools(snowpark_session)[1]]
        )
    
    coder_task = Task(
        description="""Generate Python implementation based on ALL PREVIOUS FINDINGS.
        
        Context: Use outputs from requirements analysis, sklearn research, and Streamlit research tasks.
        
        Steps:
        1. Review all previous task outputs to ensure complete requirement coverage
        2. Implement each component following the documented specifications
        3. Maintain clear traceability between code and requirements
        4. Include comments linking code sections to specific requirements""",
        expected_output="""
        1. Complete Python implementation with clear requirement traceability
        2. Code documentation referencing original requirements
        3. Implementation notes explaining how each requirement is fulfilled""",
        agent=coder_agent,
        context=[requirement_task, researcher_sklearn_task, researcher_streamlit_task]
        )
    
    tasks = [requirement_task, researcher_sklearn_task, researcher_streamlit_task, coder_task]
    
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
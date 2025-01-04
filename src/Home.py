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
        Output only a complete Python implementation code based on requirements defined""",
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

def init_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.is_generating = False
        st.session_state.generated_code = None
        st.session_state.app_results = None
        st.session_state.current_progress = 0
        st.session_state.status_message = ""

def main():
    # Initialize session state
    init_session_state()
    
    st.title("Bad Scientist: Streamlit App Builder")
    
    # Only show inputs if not currently generating
    if not st.session_state.is_generating:
        # Business idea prompt input
        prompt = st.text_area(
            "Enter your business idea prompt:", 
            help="Describe your business idea or requirements in detail"
        )
        
        # Optional document upload
        uploaded_file = st.file_uploader(
            "Upload business requirements document (optional)",
            type=['pdf', 'txt'],
            help="Upload additional documentation to provide more context"
        )
        
        if st.button("Generate App", type="primary"):
            if prompt:
                try:
                    st.session_state.is_generating = True
                    with st.spinner("Initializing generation process..."):
                        # Handle document upload
                        docs_path = None
                        if uploaded_file:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                docs_path = tmp_file.name
                        
                        # Create crew
                        crew = create_crew(prompt, docs_path)
                        
                        # Progress indicators
                        st.write("Starting generation process...")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Execute tasks
                        status_text.write("Analyzing requirements...")
                        crew.kickoff()
                        progress_bar.progress(25)
                        
                        # Get individual task results
                        requirement_task = crew.tasks[0]
                        sklearn_task = crew.tasks[1]
                        streamlit_task = crew.tasks[2]
                        coder_task = crew.tasks[3]
                        
                        # Update progress for each task
                        status_text.write("Processing scikit-learn implementation...")
                        progress_bar.progress(50)
                        
                        status_text.write("Generating Streamlit interface...")
                        progress_bar.progress(75)
                        
                        status_text.write("Creating final implementation...")
                        progress_bar.progress(100)
                        
                        # Store results in session state
                        st.session_state.generated_code = coder_task.output.raw
                        st.session_state.app_results = {
                            "requirements": requirement_task.output.raw,
                            "sklearn_specs": sklearn_task.output.raw,
                            "streamlit_specs": streamlit_task.output.raw,
                        }
                        
                        # Success message with debug info
                        st.success("Generation complete! Navigate to the Generated App page to view and run your application.")
                        
                        # Add a reset button
                        if st.button("Generate Another App"):
                            st.session_state.is_generating = False
                            st.experimental_rerun()
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)
                    st.session_state.is_generating = False
                finally:
                    # Cleanup
                    if docs_path:
                        try:
                            os.unlink(docs_path)
                        except Exception as e:
                            st.warning(f"Could not remove temporary file: {str(e)}")
            else:
                st.error("Please enter a business idea prompt")
    else:
        # Show status when generation is complete
        st.success("Generation is complete! Navigate to the Generated App page to view and run your application.")
        if st.button("Generate Another App"):
            st.session_state.is_generating = False
            st.rerun()

if __name__ == "__main__":
    main()
from typing import Optional
from crewai import Crew, Task, Process, LLM
from langchain_openai import ChatOpenAI
import os

# Import agents
from agents.requirements.requirements import RequirementAgent
from agents.researcher.researcher import ResearcherAgent
from agents.coder.coder import CoderAgent
from tools.search_cortex import create_search_tools, DocumentProcessor, DocumentType

def create_crew(prompt: str, docs_uploaded: bool, docs_path: Optional[str] = None):
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
        description=f"""Analyze the business requirements input prompt and document using the Search Requirements Documents tool.
        Input: {prompt}.
        docs_uploaded = {docs_uploaded}
        
        Steps:
        If docs_uploaded is FALSE, no need to use the tool, just analyze  technical components to implement using Python only from the input prompt.
        If docs_uploaded is TRUE:
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
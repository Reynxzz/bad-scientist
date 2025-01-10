from typing import Optional
from crewai import Crew, Task, Process
import os
from agents.requirements import RequirementAgent
from agents.researcher import ResearcherAgent
from agents.data_analyst import DataAnalysisAgent
from agents.coder import CoderAgent
from tools.search_cortex import CortexSearchRequirementsTool, CortexSearchTechnicalTool, DocumentProcessor, DocumentType
from tools.get_snowflake_tables import SnowflakeTableTool
from custom_cortex_llm.snowflake_mistral_agents import CrewSnowflakeLLM

def create_crew(prompt: str, docs_uploaded: bool, docs_path: Optional[str] = None):
    """Create and configure the agent crew for text-to-Streamlit app generation
    
    Args:
        prompt (str): User's input prompt describing desired Streamlit app
        docs_uploaded (bool): Whether additional requirement documents were uploaded
        docs_path (Optional[str]): Path to uploaded requirement documents
        
    Returns:
        Crew: Configured CrewAI instance with sequential task pipeline
    """
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

    # Initialize LLM and tools
    llm = CrewSnowflakeLLM(
        session=snowpark_session,
        model_name="mistral-large2",
        temperature=0.3,
    )

    search_req_tool = CortexSearchRequirementsTool(snowpark_session, result_as_answer=True)
    search_tech_tools = CortexSearchTechnicalTool(snowpark_session, result_as_answer=True)
    analysis_tools = SnowflakeTableTool(snowpark_session, result_as_answer=True)

    # Initialize agents
    requirement_agent = RequirementAgent(llm, [search_req_tool])
    data_agent = DataAnalysisAgent(llm, [analysis_tools])    
    researcher_agent = ResearcherAgent(llm, [search_tech_tools])
    coder_agent = CoderAgent(llm)
    
    # Process uploaded documents if available
    if docs_path:
        doc_processor = DocumentProcessor(snowpark_session)
        doc_processor.process_document(docs_path, DocumentType.REQUIREMENTS)
    
    # Task 1: Requirements Analysis
    requirement_task = Task(
        description=f"""Extract and analyze technical requirements for Streamlit app implementation.

        Input: {prompt}
        Documents uploaded: {docs_uploaded}

        Instructions:
        1. If documents are uploaded (docs_uploaded=True):
        - Use the Search Requirements Documents tool with relevant keywords from the prompt
        - Use doc_type="requirements" for the search
        - Extract technical requirements from search results
        2. If no documents (docs_uploaded=False):
        - Analyze the input prompt directly
        3. For all cases:
        - Focus ONLY on Python-implementable components
        - Identify specific Streamlit UI elements needed
        - List data processing/analysis requirements
        - Note any integration requirements (e.g., file uploads, APIs)
        - Specify any computational or algorithmic needs""",
        expected_output="""Provide a structured output with:
        1. Core Technical Requirements:
        - List of required Streamlit UI components
        - Data processing/analysis needs
        - Integration requirements
        2. Implementation Constraints:
        - Required Python libraries
        - Performance considerations
        - User interaction flows""",
        agent=requirement_agent,
        tools=[search_req_tool]
    )
    
    # Task 2: Data Analysis
    data_analysis_task = Task(
        description="""Evaluate and map Snowflake data requirements for the application.

        Instructions:
        1. Review technical requirements from previous task
        2. Determine if Snowflake data access is needed:
        - If NO: Skip to output with "No Snowflake data required"
        - If YES: Continue with steps 3-5
        3. Use 'Search Snowflake Tables' tool to identify relevant tables:
        - Craft specific search queries based on requirements
        - Example: query="Find tables related to customer transactions"
        4. For each identified table:
        - Validate column availability against requirements
        - Check data types and constraints
        5. Document exact table and column names for implementation""",
        expected_output="""Provide either:
        1. "No Snowflake data required" statement OR
        2. Detailed data mapping:
        - Exact table and column names
        - SQL queries for data access
        - Python code examples for data integration""",
        agent=data_agent,
        tools=[analysis_tools],
        context=[requirement_task]
    )
    
    # Task 3: Reference App Research
    researcher_reference_app_task = Task(
        description="""Research existing Streamlit implementations for reference.

        Instructions:
        1. For each technical component from requirements:
        - Use search_tech_tools with:
            - doc_type="technical_docs"
            - tech_stack="st_ref"
        - Search for similar implementations
        2. Analyze found references for:
        - UI patterns and layouts
        - Data handling approaches
        - User interaction patterns
        3. Map reference implementations to current requirements
        4. Identify best practices and optimization opportunities""",
        expected_output="""Provide:
        1. Relevant code patterns for each requirement
        2. Streamlit-specific implementation details
        3. Performance optimization suggestions
        4. Error handling patterns""",
        agent=researcher_agent,
        context=[requirement_task, data_analysis_task],
        tools=[search_tech_tools]
    )
    
    # Task 4: Streamlit Documentation Validation
    researcher_streamlit_task = Task(
        description="""Validate implementation patterns against current Streamlit documentation.

        Instructions:
        1. For each UI component and feature:
        - Use search_tech_tools with:
            - doc_type="technical_docs"
            - tech_stack="streamlit"
        - Verify latest Streamlit usage
        - Check for deprecated features
        3. Validate data display components
        4. Verify file handling methods""",
        expected_output="""Provide:
        1. Validated Streamlit component usage
        2. Current best practices
        3. Required import statements""",
        agent=researcher_agent,
        context=[requirement_task, data_analysis_task, researcher_reference_app_task],
        tools=[search_tech_tools]
    )
    
    # Task 5: Code Implementation
    coder_task = Task(
        description="""Generate complete, production-ready Streamlit application code.

        Instructions:
        1. Implement all components using validated patterns
        2. Include proper error handling for:
        - Data loading/processing
        - User inputs
        - API calls
        - File operations
        3. Implement consistent state management
        4. Add input validation
        5. Include performance optimizations
        6. Use Snowflake connection from .env if required

        Notes:
        - Code must be immediately runnable
        - Use actual Snowflake tables (no dummy data)
        - Assume credentials in .env file
        - Include all necessary imports""",
        expected_output="""Provide:
        1. Complete, runnable 1 Page ONLY Python/Streamlit code
        2. All required import statements
        3. Properly structured main() function. Don't use 'st.set_page_config' to prevent error
        4. Clear code organization
        5. Error handling implementation
        6. No need for code explaination, just the code itself""",
        agent=coder_agent,
        context=[requirement_task, data_analysis_task, researcher_reference_app_task, researcher_streamlit_task]
    )
    
    # Configure and return crew
    crew = Crew(
        agents=[requirement_agent, data_agent, researcher_agent, coder_agent],
        tasks=[requirement_task, data_analysis_task, researcher_reference_app_task, researcher_streamlit_task, coder_task],
        process=Process.sequential,
        verbose=True
    )
    
    return crew
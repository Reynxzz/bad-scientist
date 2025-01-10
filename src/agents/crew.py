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
    """Create and configure the agent crew"""
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

    llm = CrewSnowflakeLLM(
        session=snowpark_session,
        model_name="mistral-large2",
        temperature=0.3,
    )

    search_req_tool = CortexSearchRequirementsTool(snowpark_session, result_as_answer=True)
    search_tech_tools = CortexSearchTechnicalTool(snowpark_session, result_as_answer=True)
    analysis_tools = SnowflakeTableTool(snowpark_session, result_as_answer=True)

    requirement_agent = RequirementAgent(llm, [search_req_tool])
    data_agent = DataAnalysisAgent(llm, [analysis_tools])    
    researcher_agent = ResearcherAgent(llm, [search_tech_tools])
    coder_agent = CoderAgent(llm)
    
    if docs_path:
        doc_processor = DocumentProcessor(snowpark_session)
        doc_processor.process_document(docs_path, DocumentType.REQUIREMENTS)
    
    requirement_task = Task(
        description=f"""Analyze the business requirements input prompt and document using the Search Requirements Documents tool.
        Input: {prompt}.
        docs_uploaded = {docs_uploaded}
        
        Steps:
        If docs_uploaded is FALSE, no need to use the tool, just analyze  technical components to implement using Python only from the input prompt.
        If docs_uploaded is TRUE:
            1. Use the Search Requirements Documents tool to query relevant requirements with:
            - query: Extract key phrases from the prompt
            - doc_type: "requirements" (you can only use this parameter to use the tools)
            2. Analyze and extract key technical components to implement using Python only from the search results""",
        expected_output="""
        Detailed list of technical components and requirements to implement using Python only""",
        agent=requirement_agent,
        tools=[search_req_tool]
        )
    
    data_analysis_task = Task(
        description="""Determine if the requirements need to use available database or not. 
        If not, just skip this step. No need to use tools.
        If yes, analyze available Snowflake tables for implementation.

        Steps:
        1. Review the requirements from the previous task
        2. Use the 'Search Snowflake Tables' tool to find relevant tables by providing a search query
           Example: you can search with query="Find tables related to rides and drivers"
        3. Analyze the returned table structures and sample data
        4. Map the available tables and fields to the specific requirements""",
        expected_output="""If there Snowflake data needed or can be used:
        1. Detailed mapping between business requirements and available data fields. Don't make up any table names or columns that are not available. If not neccesary to use table, just say no data needed. 
        2. Recommendations for data usage in implementation and example how can we use it in python. If not neccesary to use table, just say no data needed.""",
        agent=data_agent,
        tools=[analysis_tools],
        context=[requirement_task]
    )
    
    researcher_reference_app_task = Task(
        description="""
        Find existing streamlit app as your reference and inspiration to build streamlit app based on REQUIREMENTS needed. 
        
        Context: Use the output from the requirements analysis and data analysis task as your foundation.
        
        Steps:
        1. For each technical component previously identified in the requirements:
        - Use the `search_tech_tools` tool with:
            - query: Component name and key technical terms from requirements
            - doc_type: "technical_docs"
            - tech_stack: "st_ref" (you can only use this parameter to use the tools)
        2. Generate a streamlit implementation pattern to fulfill app requirements""",
        expected_output="""
        Streamlit (and other supporting library) implementation code in python based on requirements and existing streamlit app reference.""",
        agent=researcher_agent,
        context=[requirement_task, data_analysis_task],
        tools=[search_tech_tools]
        )
    
    researcher_streamlit_task = Task(
        description="""Research Streamlit implementation details using the `search_tech_tools` and validate previous code based on latest streamlit documentations.
        
        Context: Use outputs from previous requirements, data analysis (if available), and reference app research tasks.
        
        Steps:
        1. For each UI component needed to fulfill the original requirements:
        - Use the `search_tech_tools` tool to search latest streamlit documentation with:
            - query: Component requirements and sklearn integration points
            - doc_type: "technical_docs" (you can only use this parameter to use the tools)
            - tech_stack: "streamlit" (you can only use this parameter to use the tools)
        2. Ensure UI patterns align with both business requirements and latest streamlit documentation implementation.""",
        expected_output="""
        Streamlit's UI implementation code that fulfill the requirements and current streamlit documentation.""",
        agent=researcher_agent,
        context=[requirement_task, data_analysis_task, researcher_reference_app_task],
        tools=[search_tech_tools]
        )
    
    coder_task = Task(
        description="""Generate Python implementation based on ALL PREVIOUS FINDINGS.
        
        Context: Use outputs from requirements analysis, data analysis, reference app research, and Streamlit research tasks as guidance.
        
        Steps:
        1. Review all previous task outputs to ensure complete requirement coverage
        2. Implement each component in python and streamlit. 
        Please note that the expected output is runnable streamlit app that users just can run it directly, don't use example or dummy data if Snowflake table are needed.
        Obtained and assume all secrets and credentials there in .env file.""",
        expected_output="""
        Output only a complete Python/streamlit implementation runnable and working code directly. No need to add explaination or anything other than python code.""",
        agent=coder_agent,
        context=[requirement_task, data_analysis_task, researcher_reference_app_task, researcher_streamlit_task],
        )
    
    tasks = [requirement_task, data_analysis_task, researcher_reference_app_task, researcher_streamlit_task, coder_task]
    
    crew = Crew(
        agents=[requirement_agent, data_agent, researcher_agent, coder_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )
    
    return crew
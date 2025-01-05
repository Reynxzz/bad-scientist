from typing import Optional
from crewai import Crew, Task, Process, LLM
from langchain_openai import ChatOpenAI
import os

# Import agents
from agents.requirements import RequirementAgent
from agents.researcher import ResearcherAgent
from agents.data_analyst import DataAnalysisAgent
from agents.coder import CoderAgent
from tools.search_cortex import create_search_tools, DocumentProcessor, DocumentType
from tools.get_snowflake_tables import SnowflakeTableTool
from custom_cortex_llm.snowflake_mistral_agents import CrewSnowflakeLLM

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

    llm = CrewSnowflakeLLM(
        session=snowpark_session,
        model_name="mistral-large2",
        temperature=0.5,
    )

    # llm = LLM(
    #     model="gemini/gemini-1.5-flash",
    #     temperature=0.7
    # )
    
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Create search tools
    search_tools = create_search_tools(snowpark_session)
    analysis_tools = SnowflakeTableTool(snowpark_session)

    # Initialize agents
    requirement_agent = RequirementAgent(llm, search_tools)
    data_agent = DataAnalysisAgent(llm, [analysis_tools])    
    researcher_agent = ResearcherAgent(llm, search_tools)
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
            - doc_type: "requirements" (you can only use this parameter to use the tools)
            2. Analyze and extract key technical components to implement using Python only from the search results""",
        expected_output="""
        Detailed list of technical components and requirements to implement using Python only""",
        agent=requirement_agent,
        tools=[create_search_tools(snowpark_session)[0]]
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
        expected_output="""Provide a detailed analysis report containing:
        1. Detailed mapping between business requirements and available data fields. Don't make up any table names or columns that are not available. If not neccesary to use table, just say no data needed. 
        2. Recommendations for data usage in implementation and example how can we use it in python. If not neccesary to use table, just say no data needed.""",
        agent=data_agent,
        tools=[analysis_tools],
        context=[requirement_task]
    )
    
    tech_research_task = Task(
        description="""Based on requirements:
            1. If ML needed: Research scikit-learn implementation
            2. Design Streamlit UI components
            3. Identify additional Python libraries needed
            
        Search technical docs one at a time with:
            - ML components: tech_stack='sklearn'
            - UI components: tech_stack='streamlit'""",
        agent=researcher_agent,
        expected_output="""
        Output only a complete Python/streamlit/scikit-learn implementation code.""",
        tools=[create_search_tools(snowpark_session)[1]],
        context=[requirement_task, data_analysis_task]
    )
    
    coder_task = Task(
        description="""Generate Python implementation based on ALL PREVIOUS FINDINGS.
        
        Context: Use outputs from requirements analysis, data analysis, sklearn research, and Streamlit research tasks as guidance.
        
        Steps:
        1. Review all previous task outputs to ensure complete requirement coverage
        2. Implement each component in python and streamlit""",
        expected_output="""
        Output only a complete Python/streamlit implementation code. No need to add explanation or anything other than python code""",
        agent=coder_agent,
        context=[requirement_task, data_analysis_task, tech_research_task]
        )
    
    tasks = [requirement_task, data_analysis_task, tech_research_task, coder_task]
    
    # Create crew
    crew = Crew(
        agents=[requirement_agent, data_agent, researcher_agent, coder_agent], #validator_agent
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )
    
    return crew
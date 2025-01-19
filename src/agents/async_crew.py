from typing import Optional, Dict, Any
import asyncio
from pydantic import BaseModel
from crewai import Crew, Task
from config import CONNECTION_PARAMETER, MODEL_NAME, MODEL_TEMPERATURE
from agents.requirements import RequirementAgent
from agents.researcher import ResearcherAgent
from agents.data_analyst import DataAnalysisAgent
from agents.coder import CoderAgent
from tools.search_cortex import CortexSearchRequirementsTool, CortexSearchTechnicalTool, DocumentType
from tools.document_processor import DocumentProcessor
from tools.get_snowflake_tables import SnowflakeTableTool
from custom_cortex_llm.snowflake_mistral_agents import CrewSnowflakeLLM
from snowflake.snowpark.session import Session
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FlowResult(BaseModel):
    """Result model for flow execution"""
    requirements: str = ""
    data_analysis: str = ""
    reference_patterns: Dict[str, Any] = {}
    streamlit_components: str = ""
    final_code: str = ""

async def execute_streamlit_generation(
    prompt: str,
    docs_uploaded: bool = False,
    docs_path: Optional[str] = None
) -> FlowResult:
    """
    Execute the Streamlit app generation process using multiple async crews
    
    Args:
        prompt (str): User's request for the Streamlit app
        docs_uploaded (bool): Whether additional documents were uploaded
        docs_path (Optional[str]): Path to uploaded documents
        
    Returns:
        FlowResult: Complete results from flow execution
    """
    try:
        # Initialize shared resources
        logger.info("Initializing shared resources...")
        snowpark_session = Session.builder.configs(CONNECTION_PARAMETER).create()
        llm = CrewSnowflakeLLM(
            session=snowpark_session,
            model_name=MODEL_NAME,
            temperature=MODEL_TEMPERATURE
        )

        # Process documents if provided
        if docs_uploaded and docs_path:
            logger.info(f"Processing uploaded document: {docs_path}")
            doc_processor = DocumentProcessor(snowpark_session)
            doc_processor.process_document(docs_path, DocumentType.REQUIREMENTS)

        # Initialize tools
        logger.info("Initializing tools...")
        search_req_tool = CortexSearchRequirementsTool(snowpark_session, result_as_answer=True)
        search_tech_tool = CortexSearchTechnicalTool(snowpark_session, result_as_answer=True)
        analysis_tool = SnowflakeTableTool(snowpark_session, result_as_answer=True)

        # Create agents
        logger.info("Creating agents...")
        requirement_agent = RequirementAgent(llm, [search_req_tool])
        data_agent = DataAnalysisAgent(llm, [analysis_tool])
        researcher_agent = ResearcherAgent(llm, [search_tech_tool])
        # coder_agent = CoderAgent(llm)

        # Create Requirements Analysis Crew
        logger.info("Setting up requirements analysis...")
        requirements_task = Task(
            description=f"""
            OBJECTIVE: Extract and analyze comprehensive technical requirements for a Streamlit application.

            CONTEXT:
            User Request: {prompt}
            Additional Documents: {'Yes' if docs_uploaded else 'No'}

            TASK STEPS:
            Review Input Sources:
               - Analyze the main user prompt
               - If documents are provided, use Search Requirements tool
               - Search with keywords from prompt using the tools with specific parameter doc_type="requirements"
            """,
            expected_output="""A structured breakdown of technical requirements including:
            1. Technical Requirements
            2. Data Requirements
            3. UI Components
            4. Integration Needs
            5. Constraints""",
            agent=requirement_agent
        )

        requirements_crew = Crew(
            agents=[requirement_agent],
            tasks=[requirements_task]
        )

        # Execute requirements analysis first
        logger.info("Starting requirements analysis")
        requirements_output = await requirements_crew.kickoff_async(inputs={})
        requirements_result = str(requirements_output.raw)
        logger.info("Requirements analysis complete")
        
        # Create parallel crews for data analysis and research
        logger.info("Setting up data analysis and research tasks...")
        data_task = Task(
            description=f"""
            OBJECTIVE: Extract tables needed to fulfill the requirements and call the Snowflake Table Tools by passing the keywords (3-5 keywords).
            
            CONTEXT:
            Technical Requirements: {requirements_result}
            
            TASK STEPS:
            1. Before calling the function, decide is the requirements needed data from Snowflake or not. If no related/relevant tables needed or can be usec. Just simply response: "No Snowflake data required".
            2. Use Snowflake table tools to analyze Snowflake data needed based on requirements. Pass the detailed query to the function.
            """,
            expected_output="""Detailed data mapping including:
            1. Required Tables
            2. Required Columns
            3. Sample SQL Queries in Streamlit""",
            agent=data_agent
        )

        patterns_task = Task(
            description=f"""
            OBJECTIVE: Extract Streamlit data visualization component to fulfill the requirements and call the Search Technical Tools by passing the keywords (3-5 keywords) in query and one more parameter, tech_stack = 'st_ref'.
            
            CONTEXT:
            Technical Requirements: {requirements_result}
            
            TASK STEPS:
            Research Streamlit Data Visualization, Interaction, Snowflake Data Loading and code reference using Search Technical Tools with passing parameter tech_stack = 'st_ref' (only use this parameter for this task)
            """,
            expected_output="""Comprehensive pattern guide including:
            1. Visualization Patterns
            2. Interaction Patterns
            3. Data Loading Patterns
            4. Code References""",
            agent=researcher_agent
        )

        data_crew = Crew(
            agents=[data_agent],
            tasks=[data_task]
        )

        research_crew = Crew(
            agents=[researcher_agent],
            tasks=[patterns_task]
        )

        # Execute data analysis and research crews in parallel
        logger.info("Starting parallel data analysis and research")
        data_future = data_crew.kickoff_async(inputs={})
        research_future = research_crew.kickoff_async(inputs={})
        
        # Wait for both crews to complete
        data_output, patterns_output = await asyncio.gather(data_future, research_future)
        data_result = str(data_output.raw)
        patterns_result = str(patterns_output.raw)
        logger.info("Parallel tasks complete")

        # with open('./agents/streamlit_template.txt', 'r') as file:
        #     streamlit_template = file.read() 
        

        code_result = llm.call(f"""
            OBJECTIVE: Generate production-ready python Streamlit application code based to connect and visualize Snowflake data.
            Use visualization library like plotly or matplotlib to make engaging data visualization.
            
            INPUTS:
            Data Analysis: {data_result}
            
            STEPS:
            Only outputing PYTHON code, no need any explanation, just the code.
            Assume all credentials are stored in .env
        """)
        # # Create and execute final code generation crew
        # logger.info("Setting up code generation...")
        # code_task = Task(
        #     description=f"""
        #     OBJECTIVE: Generate production-ready Streamlit application code based on streamlit template.
            
        #     INPUTS:
        #     1. Requirements: {requirements_result}
        #     2. Data Analysis: {data_result}
        #     3. Patterns: {patterns_result}
            
        #     DEVELOPMENT REQUIREMENTS:
        #     Use this streamlit app template but modify based on requirements and dataand visualization needed. Assume all credentials are stored in .env
        #     Streamlit Template: {strealit_template}
        #     """,
        #     expected_output="""Complete, production-ready Streamlit application code including:
        #     1. Import statements
        #     2. Configuration and setup
        #     3. Helper functions
        #     4. Main application code
        #     5. Error handling implementations""",
        #     agent=coder_agent
        # )

        # code_crew = Crew(
        #     agents=[coder_agent],
        #     tasks=[code_task]
        # )

        # logger.info("Starting code generation")
        # code_output = await code_crew.kickoff_async(inputs={})
        # code_result = str(code_output.raw)
        # logger.info("Code generation complete")

        # Compile results
        logger.info("Compiling final results")
        return FlowResult(
            requirements=requirements_result,
            data_analysis=data_result,
            reference_patterns={"data": patterns_result},
            streamlit_components='',
            final_code=code_result
        )

    except Exception as e:
        logger.error(f"Error in flow execution: {str(e)}")
        raise

# async def main():
#     # Example prompt
#     sample_prompt = """
#     Create a Streamlit dashboard for sales analysis with:
#     1. Monthly revenue trends
#     2. Top-performing products
#     3. Regional sales breakdown
#     Include interactive filters and make it user-friendly.
#     """
    
#     try:
#         result = await execute_streamlit_generation(
#             prompt=sample_prompt,
#             docs_uploaded=False
#         )
        
#         # Print results
#         print("\nRequirements Analysis:")
#         print(result.requirements)
#         print("\nData Analysis:")
#         print(result.data_analysis)
#         print("\nReference Patterns:")
#         print(result.reference_patterns)
#         print("\nStreamlit Components:")
#         print(result.streamlit_components)
#         print("\nGenerated Code:")
#         print(result.final_code)
        
#     except Exception as e:
#         logger.error(f"Main execution failed: {str(e)}")
#         raise

# if __name__ == "__main__":
#     asyncio.run(main())
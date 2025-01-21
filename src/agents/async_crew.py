from typing import Optional, Dict, Any
import asyncio
from pydantic import BaseModel
from crewai import Crew, Task
from config import CONNECTION_PARAMETER, MODEL_NAME, MODEL_TEMPERATURE
from agents.requirements import RequirementAgent
from agents.researcher import ResearcherAgent
from agents.data_analyst import DataAnalysisAgent
from agents.coder import CoderAgent
from tools.search_cortex import CortexSearchRequirementsTool, CortexSearchTechnicalTool
from tools.text_to_sql import SnowflakeTableTool
from tools.text_to_sklearn import SklearnImplementationTool
from tools.text_to_matplotlib import MatplotlibVisualizationTool
from custom_cortex_llm.snowflake_mistral_agents import CrewSnowflakeLLM
from snowflake.snowpark.session import Session
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedFlowResult(BaseModel):
    """Enhanced result model for flow execution"""
    requirements: str = ""
    data_analysis: str = ""
    sklearn_implementation: str = ""
    visualization_code: str = ""
    reference_patterns: Dict[str, Any] = {}
    streamlit_app: str = ""

async def execute_enhanced_generation(
    prompt: str,
    docs_uploaded: bool = False,
    docs_path: Optional[str] = None
) -> EnhancedFlowResult:
    """
    Execute enhanced generation process including Streamlit, sklearn, and visualization
    
    Args:
        prompt (str): User's request
        docs_uploaded (bool): Whether additional documents were uploaded
        docs_path (Optional[str]): Path to uploaded documents
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

        # Initialize tools
        logger.info("Initializing tools...")
        search_req_tool = CortexSearchRequirementsTool(snowpark_session)
        search_tech_tool = CortexSearchTechnicalTool(snowpark_session)
        snowflake_tool = SnowflakeTableTool(snowpark_session)
        sklearn_tool = SklearnImplementationTool(snowpark_session)
        viz_tool = MatplotlibVisualizationTool(snowpark_session)

        # Create agents with enhanced toolsets
        logger.info("Creating enhanced agents...")
        requirement_agent = RequirementAgent(llm, [search_req_tool])
        data_agent = DataAnalysisAgent(llm, [snowflake_tool])
        researcher_agent = ResearcherAgent(llm, [search_tech_tool])
        ml_agent = DataAnalysisAgent(llm, [sklearn_tool])
        viz_agent = DataAnalysisAgent(llm, [viz_tool])

        # Requirements Analysis Task
        requirements_task = Task(
            description=f"""
            Analyze requirements for data science application including:
            - Data analysis needs
            - Machine learning requirements
            - Visualization requirements
            
            User Request: {prompt}
            """,
            expected_output="""Structured requirements document containing:
            1. Data Analysis Requirements
            2. Machine Learning Requirements
            3. Visualization Requirements
            4. Integration Points
            5. Technical Constraints""",
            agent=requirement_agent
        )

        # Execute requirements analysis
        requirements_crew = Crew(
            agents=[requirement_agent],
            tasks=[requirements_task]
        )
        requirements_output = await requirements_crew.kickoff_async()
        requirements_result = str(requirements_output.raw)

        # Parallel Data Analysis Tasks
        data_task = Task(
            description=f"""
            Analyze required Snowflake data based on:
            {requirements_result}
            """,
            expected_output="""Detailed data analysis including:
            1. Required Snowflake Tables
            2. SQL Queries
            3. Data Schema
            4. Data Relationships
            5. Sample Data Format""",
            agent=data_agent
        )

        sklearn_task = Task(
            description=f"""
            Generate sklearn implementation based on:
            {requirements_result}
            """,
            expected_output="""Complete sklearn implementation including:
            1. Data Preprocessing Code
            2. Model Selection
            3. Training Pipeline
            4. Evaluation Metrics
            5. Prediction Functions""",
            agent=ml_agent
        )

        viz_task = Task(
            description=f"""
            Generate visualization code based on:
            {requirements_result}
            """,
            expected_output="""Complete visualization implementation including:
            1. Data Preparation for Plotting
            2. Matplotlib/Seaborn Visualizations
            3. Interactive Components
            4. Layout Configuration
            5. Style Customization""",
            agent=viz_agent
        )

        # Execute parallel analysis tasks
        data_crew = Crew(agents=[data_agent], tasks=[data_task])
        sklearn_crew = Crew(agents=[ml_agent], tasks=[sklearn_task])
        viz_crew = Crew(agents=[viz_agent], tasks=[viz_task])

        data_future = data_crew.kickoff_async()
        sklearn_future = sklearn_crew.kickoff_async()
        viz_future = viz_crew.kickoff_async()

        data_output, sklearn_output, viz_output = await asyncio.gather(
            data_future, sklearn_future, viz_future
        )

        # Generate final Streamlit application
        streamlit_code = llm.call(f"""
            Generate production-ready Streamlit application that integrates:
            
            1. Data Analysis: {str(data_output.raw)}
            2. Machine Learning: {str(sklearn_output.raw)}
            3. Visualizations: {str(viz_output.raw)}
            
            Only output PYTHON code, no explanations.
            Assume credentials in .env file.
        """)

        return EnhancedFlowResult(
            requirements=requirements_result,
            data_analysis=str(data_output.raw),
            sklearn_implementation=str(sklearn_output.raw),
            visualization_code=str(viz_output.raw),
            streamlit_app=streamlit_code
        )

    except Exception as e:
        logger.error(f"Error in enhanced flow: {str(e)}")
        raise

# Example usage
async def main():
    prompt = """
    Create a data science application that:
    1. Analyzes sales data from Snowflake
    2. Implements predictive analytics for future sales
    3. Visualizes trends and predictions
    4. Provides interactive dashboard in Streamlit
    """
    
    result = await execute_enhanced_generation(prompt)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
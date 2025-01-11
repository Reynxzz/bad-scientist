# crew.py
from typing import Optional
from pydantic import BaseModel
from crewai import Task, Flow
import os
from agents.requirements import RequirementAgent
from agents.researcher import ResearcherAgent
from agents.data_analyst import DataAnalysisAgent
from agents.coder import CoderAgent
from tools.search_cortex import CortexSearchRequirementsTool, CortexSearchTechnicalTool, DocumentProcessor, DocumentType
from tools.get_snowflake_tables import SnowflakeTableTool
from custom_cortex_llm.snowflake_mistral_agents import CrewSnowflakeLLM
from snowflake.snowpark.session import Session
from dotenv import load_dotenv
from crewai.flow.flow import start, listen, router, and_, or_
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class FlowResult(BaseModel):
    """Result model for flow execution"""
    requirements: str = ""
    data_analysis: str = ""
    reference_patterns: dict = {}
    streamlit_components: str = ""
    final_code: str = ""

class StreamlitAppGenerationFlow(Flow):
    def __init__(self, prompt: str, docs_uploaded: bool, docs_path: Optional[str] = None):
        super().__init__()
        logger.debug(f"Initializing flow with prompt: {prompt}")
        self.prompt = prompt
        self.docs_uploaded = docs_uploaded
        self.docs_path = docs_path
        self.result = FlowResult()
        self._initialize_resources()

    def _initialize_resources(self):
        """Initialize Snowflake, LLM, tools and agents"""
        try:
            logger.debug("Initializing Snowflake connection")
            load_dotenv()
            connection_params = {
                "account": os.getenv("SNOWFLAKE_ACCOUNT"),
                "user": os.getenv("SNOWFLAKE_USER"),
                "password": os.getenv("SNOWFLAKE_USER_PASSWORD"),
                "role": os.getenv("SNOWFLAKE_ROLE"),
                "database": os.getenv("SNOWFLAKE_DATABASE"),
                "schema": os.getenv("SNOWFLAKE_SCHEMA"),
                "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE")
            }
            self.snowpark_session = Session.builder.configs(connection_params).create()
            
            logger.debug("Initializing LLM")
            self.llm = CrewSnowflakeLLM(
                session=self.snowpark_session,
                model_name="mistral-large2",
                temperature=0.3
            )
            
            logger.debug("Initializing tools")
            self.search_req_tool = CortexSearchRequirementsTool(self.snowpark_session, result_as_answer=True)
            self.search_tech_tool = CortexSearchTechnicalTool(self.snowpark_session, result_as_answer=True)
            self.analysis_tool = SnowflakeTableTool(self.snowpark_session, result_as_answer=True)
            
            logger.debug("Initializing agents")
            self.requirement_agent = RequirementAgent(self.llm, [self.search_req_tool])
            self.data_agent = DataAnalysisAgent(self.llm, [self.analysis_tool])
            self.researcher_agent = ResearcherAgent(self.llm, [self.search_tech_tool])
            self.coder_agent = CoderAgent(self.llm)
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise

    @start()
    def process_requirements(self):
        """Initial task to analyze requirements"""
        logger.debug("Starting requirements analysis")
        try:
            if self.docs_uploaded and self.docs_path:
                logger.debug(f"Processing document: {self.docs_path}")
                doc_processor = DocumentProcessor(self.snowpark_session)
                doc_processor.process_document(self.docs_path, DocumentType.REQUIREMENTS)
            
            task = Task(
                description=f"""Extract and analyze technical requirements for Streamlit app implementation.

                Input: {self.prompt}
                Documents uploaded: {self.docs_uploaded}

                Instructions:
                1. If documents are uploaded (docs_uploaded=True):
                - Use the Search Requirements Documents tool with relevant keywords from the prompt
                - Use doc_type="requirements" for the search
                - Extract technical requirements from search results
                2. If no documents (docs_uploaded=False):
                - Analyze the input prompt directly
                3. For all cases:
                - Focus ONLY on Python-implementable components
                - List possible data needed and processing/analysis requirements
                - Identify specific Streamlit UI elements needed""",
                expected_output="""Core technical requirements, data needed, and implementation constraints""",
                agent=self.requirement_agent
            )
            
            logger.debug("Executing requirements task")
            result = task.agent.execute_task(task)
            logger.debug(f"Requirements result: {result}")
            
            self.result.requirements = result
            return result
        except Exception as e:
            logger.error(f"Error in process_requirements: {str(e)}")
            raise

    @listen(process_requirements)
    def analyze_data_needs(self, requirements):
        """Analyze data requirements based on technical specifications"""
        logger.debug("Starting data needs analysis")
        try:
            task = Task(
                description=f"""Evaluate and map Snowflake data requirements based on: {requirements}
                Evaluate and map Snowflake data requirements for the application.

                Instructions:
                1. Review technical requirements from previous task
                2. Determine if Snowflake data access is needed:
                - If NO: Skip to output with "No Snowflake data required"
                - If YES: Continue with steps 3-5
                3. Use 'Search Snowflake Tables' tool to identify relevant tables:
                - Craft specific search queries based on requirements
                - Example: query="Find tables related to customer transactions""",
                expected_output="""Provide either:
                1. "No Snowflake data required" statement OR
                2. Detailed data mapping:
                - Exact table and column names
                - SQL queries for data access
                - Python code examples for data integration""",
                agent=self.data_agent
            )
            result = task.agent.execute_task(task)
            logger.debug(f"Data analysis result: {result}")
            
            self.result.data_analysis = result
            return result
        except Exception as e:
            logger.error(f"Error in analyze_data_needs: {str(e)}")
            raise

    @listen(analyze_data_needs)
    def research_data_patterns(self, data_analysis):
        """Research data analysis patterns"""
        logger.debug("Starting data analysis patterns research")
        try:
            task = Task(
                description=f"""Research existing snowflake data analysis and visualization patterns in Streamlit and make sure the Snowflake integration based on 
                - data needed: {data_analysis} and requirements.
                You can use using 'search_tech_tool' tools to see streamlit reference app as guidance, use 'st_ref' on tech_stack parameter.""",          
                expected_output="Snowflake in Streamlit Data integration patterns, example code, and best practices",
                agent=self.researcher_agent
            )
            patterns = task.agent.execute_task(task)
            logger.debug(f"Data patterns result: {patterns}")
            
            self.result.reference_patterns["data"] = patterns
            return patterns
        except Exception as e:
            logger.error(f"Error in research_data_patterns: {str(e)}")
            raise

    @listen(research_data_patterns)
    def validate_streamlit_components(self, patterns):
        """Validate Streamlit component usage"""
        logger.debug("Starting component validation")
        try:
            task = Task(
                description=f"""Validate Streamlit implementation code using 'search_tech_tool' (pass 'streamlit' as tech_stack parameter) tools to search streamlit 
                documentation to make sure it is using latest syntax and best practice.
                
                Validate this:
                {patterns}""",
                expected_output="Validated Streamlit code component usage and best practices to fulfill the requirements align with streamlit latest documentation",
                agent=self.researcher_agent
            )
            components = task.agent.execute_task(task)
            logger.debug(f"Component validation result: {components}")
            
            self.result.streamlit_components = components
            return components
        except Exception as e:
            logger.error(f"Error in validate_streamlit_components: {str(e)}")
            raise

    @listen(validate_streamlit_components)
    def generate_final_code(self, components):
        """Generate the final Streamlit application code"""
        logger.debug("Starting code generation")
        try:
            task = Task(
                description=f"""Generate 1 page complete Streamlit application code based on:
                Streamlit Components used: {components}
                Requirements: {self.result.requirements}, please also provide error handler since you will generate a production ready streamlit code. Do not use any dummy or example data/function/component.
                Data Analysis: {self.result.data_analysis}. 
                Make sure no error when accessing or analysing data:
                - avoid case-sensitive error (example: when selecting value please use lower() function), 
                - avoid data type error (example: make sure to convert datetime to string etc.),
                - avoid unclean data error (example: make sure to fillna() null values, etc.),
                - Do not make any columns or tables other than I mentioned earlier.
                - avoid any common error that will make the app don't work.
                NOTE: if there streamlit/Snowflake authentication/credentials needed, assume it already stored in .env file""",
                expected_output="Complete, 1-page runnable error-free production-ready Python/Streamlit code. Assume all auth/credentials already stored in .env. Please always prioritize to use error handler",
                agent=self.coder_agent
            )
            code = task.agent.execute_task(task)
            logger.debug(f"Generated code length: {len(code) if code else 0}")
            
            self.result.final_code = code
            return self.result
        except Exception as e:
            logger.error(f"Error in generate_final_code: {str(e)}")
            raise

def create_flow(prompt: str, docs_uploaded: bool, docs_path: Optional[str] = None) -> StreamlitAppGenerationFlow:
    """Create and configure the Streamlit app generation flow"""
    logger.debug("Creating new flow instance")
    return StreamlitAppGenerationFlow(prompt, docs_uploaded, docs_path)
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
                Documents uploaded: {self.docs_uploaded}""",
                expected_output="""Core technical requirements and implementation constraints""",
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
                description=f"Evaluate and map Snowflake data requirements based on: {requirements}",
                expected_output="Data mapping or No Snowflake data required statement",
                agent=self.data_agent
            )
            result = task.agent.execute_task(task)
            logger.debug(f"Data analysis result: {result}")
            
            self.result.data_analysis = result
            return result
        except Exception as e:
            logger.error(f"Error in analyze_data_needs: {str(e)}")
            raise

    @router(analyze_data_needs)
    def determine_research_path(self):
        """Route to appropriate research path based on data analysis"""
        logger.debug("Determining research path")
        try:
            if "No Snowflake data required" in str(self.result.data_analysis):
                logger.debug("Selected UI research path")
                return "ui_research"
            logger.debug("Selected data research path")
            return "data_research"
        except Exception as e:
            logger.error(f"Error in determine_research_path: {str(e)}")
            raise

    @listen("data_research")
    def research_data_patterns(self, data_analysis):
        """Research data handling patterns"""
        logger.debug("Starting data patterns research")
        try:
            task = Task(
                description=f"Research optimal data handling patterns based on: {data_analysis}",
                expected_output="Data integration patterns and best practices",
                agent=self.researcher_agent
            )
            patterns = task.agent.execute_task(task)
            logger.debug(f"Data patterns result: {patterns}")
            
            self.result.reference_patterns["data"] = patterns
            return patterns
        except Exception as e:
            logger.error(f"Error in research_data_patterns: {str(e)}")
            raise

    @listen("ui_research")
    def research_ui_patterns(self, _):
        """Research UI implementation patterns"""
        logger.debug("Starting UI patterns research")
        try:
            task = Task(
                description="Research Streamlit UI patterns and best practices",
                expected_output="UI implementation patterns and best practices",
                agent=self.researcher_agent
            )
            patterns = task.agent.execute_task(task)
            logger.debug(f"UI patterns result: {patterns}")
            
            self.result.reference_patterns["ui"] = patterns
            return patterns
        except Exception as e:
            logger.error(f"Error in research_ui_patterns: {str(e)}")
            raise

    @listen(or_("research_data_patterns", "research_ui_patterns"))
    def validate_streamlit_components(self, patterns):
        """Validate Streamlit component usage"""
        logger.debug("Starting component validation")
        try:
            task = Task(
                description=f"Validate implementation patterns: {patterns}",
                expected_output="Validated component usage and best practices",
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
                description=f"""Generate complete Streamlit application code based on:
                Components: {components}
                Requirements: {self.result.requirements}
                Data Analysis: {self.result.data_analysis}""",
                expected_output="Complete, runnable Python/Streamlit code",
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
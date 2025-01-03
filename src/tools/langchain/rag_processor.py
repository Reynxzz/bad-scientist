"""
Main RAG processing module for generating applications.
"""

import os
import json
from typing import Dict
from langchain.tools import Tool
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
from snowflake.core import Root

from tools.langchain.snowflake_connection import SnowflakeConnection
from tools.langchain.search_service import SearchService
from agents.langchain.agents import LangChainAgent, AgentPrompts

class RAGProcessor:
    """Main processor for RAG-based application generation."""
    
    def __init__(self):
        # Initialize Snowflake connection
        snowflake_conn = SnowflakeConnection()
        self.session = snowflake_conn.create_session()
        self.root = Root(self.session)
        
        # Initialize search service
        self.search_service = SearchService(self.root, self.session)
        
        # Initialize LLM
        self.chat_model = ChatMistralAI(
            mistral_api_key=os.environ["MISTRAL_API_KEY"],
            model="mistral-small-latest",
            temperature=0.7
        )
        
        # Initialize agents
        self.requirements_agent = self._setup_requirements_agent()
        self.sklearn_agent = self._setup_sklearn_agent()
        self.streamlit_agent = self._setup_streamlit_agent()
    
    def _setup_requirements_agent(self) -> LangChainAgent:
        """Configure the requirements analysis agent."""
        tools = [
            Tool(
                name="SearchRequirements",
                func=lambda q: json.dumps(
                    self.search_service.search_documents(
                        "req_docs_search_svc", q, ["doc_text", "source"]
                    )
                ),
                description="Search through business requirements documents"
            )
        ]
        return LangChainAgent(self.chat_model, tools, AgentPrompts.get_requirements_prompt())
    
    def _setup_sklearn_agent(self) -> LangChainAgent:
        """Configure the scikit-learn research agent."""
        tools = [
            Tool(
                name="SearchTechnicalDocs",
                func=lambda q: json.dumps(
                    self.search_service.search_documents(
                        "sklearn_docs_search_svc", q, ["doc_text"]
                    )
                ),
                description="Search through scikit-learn documentation"
            )
        ]
        return LangChainAgent(self.chat_model, tools, AgentPrompts.get_sklearn_prompt())
    
    def _setup_streamlit_agent(self) -> LangChainAgent:
        """Configure the Streamlit research agent."""
        tools = [
            Tool(
                name="SearchTechnicalDocs",
                func=lambda q: json.dumps(
                    self.search_service.search_documents(
                        "streamlit_docs_search_svc", q, ["doc_text"]
                    )
                ),
                description="Search through Streamlit documentation"
            )
        ]
        return LangChainAgent(self.chat_model, tools, AgentPrompts.get_streamlit_prompt())
    
    def process_requirements(self, prompt: str) -> Dict[str, str]:
        """
        Process requirements and generate implementation specifications.
        
        Args:
            prompt: Business requirements prompt
            
        Returns:
            Dictionary containing processed results
        """
        # Analyze requirements
        requirements = self.requirements_agent.run(f"Analyze these requirements: {prompt}")
        
        # Research implementations
        sklearn_specs = self.sklearn_agent.run(
            f"Research sklearn implementations for these requirements: {requirements}"
        )
        
        streamlit_specs = self.streamlit_agent.run(
            f"""Research Streamlit implementations that integrate with these sklearn specs: 
            {sklearn_specs}. Original requirements: {requirements}"""
        )
        
        # Generate final implementation
        implementation = self._generate_implementation(
            requirements, sklearn_specs, streamlit_specs
        )
        
        return {
            "requirements": requirements,
            "sklearn_specs": sklearn_specs,
            "streamlit_specs": streamlit_specs,
            "implementation": implementation
        }
    
    def _generate_implementation(self, requirements: str, 
                               sklearn_specs: str, 
                               streamlit_specs: str) -> str:
        """Generate the final Python implementation."""
        prompt = PromptTemplate.from_template(
            """You are a Python developer. Generate a complete implementation that integrates 
            Streamlit UI with sklearn components. Include clear documentation linking code to 
            requirements.
            
            Requirements: {requirements}
            Sklearn Specifications: {sklearn_specs}
            Streamlit Specifications: {streamlit_specs}
            """
        )
        
        response = self.chat_model.invoke(prompt.format(
            requirements=requirements,
            sklearn_specs=sklearn_specs,
            streamlit_specs=streamlit_specs
        ))
        
        return response.content
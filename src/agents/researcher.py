from crewai import Agent
from langchain.chat_models.base import BaseChatModel
from langchain.tools import Tool
from agents.base_model import TechnicalSpec, RequirementAnalysis
from typing import List

class ResearcherAgent(Agent):
    """Agent responsible for researching technical implementations"""
    
    def __init__(self, llm: BaseChatModel, tools: List[Tool]):
        super().__init__(
            role="Technical Researcher",
            goal="""Research and provide implementation details for technical components 
            using technical documentation and best practices""",
            backstory="""Specialized in researching technical solutions and best practices. 
            You have extensive experience in finding and evaluating technical approaches 
            for complex systems. You focus on official documentation and proven python
            implementation patterns.""",
            llm=llm,
            tools=tools,
            verbose=True,
            memory=True
        )
        
    def research_implementation(self, requirements: RequirementAnalysis) -> TechnicalSpec:
        """
        Research implementation details for given requirements
        
        Args:
            requirements (RequirementAnalysis): Analyzed requirements to research
            
        Returns:
            TechnicalSpec: Detailed technical specifications
        """
        # The actual research will be handled by CrewAI's task system
        # This method is for documentation and type hinting
        pass
        
    def _research_workflow(self):
        """Define the research workflow"""
        return [
            ("Review Requirements", "Use search_requirements to understand the context"),
            ("Research Components", "Use search_technical_docs to find python implementation patterns"),
            ("Technical Analysis", "Evaluate and select appropriate solutions"),
            ("Integration Planning", "Plan how components work together"),
            ("Documentation", "Prepare technical specifications")
        ]
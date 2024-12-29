from crewai import Agent
from langchain.chat_models.base import BaseChatModel
from agents.base_model import RequirementAnalysis

class RequirementAgent(Agent):
    """Agent responsible for analyzing business requirements"""
    
    def __init__(self, llm: BaseChatModel):
        super().__init__(
            role="Requirement Analyzer",
            goal="Analyze business requirements and extract key technical components",
            backstory="""Expert at analyzing business requirements and breaking them down 
            into technical components. You have years of experience in translating business 
            needs into actionable technical specifications in python.""",
            llm=llm,
            verbose=True
        )
        
    def analyze_requirements(self, prompt: str) -> RequirementAnalysis:
        """
        Analyze the business requirements and extract key components
        
        Args:
            prompt (str): The business requirement prompt
            
        Returns:
            RequirementAnalysis: Structured analysis of the requirements
        """
        # The actual analysis will be handled by CrewAI's task system
        # This method is for documentation and type hinting
        pass
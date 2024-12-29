from crewai import Agent
from langchain.chat_models.base import BaseChatModel
from agents.base_model import TechnicalSpec

class CoderAgent(Agent):
    """Agent responsible for generating code implementations"""
    
    def __init__(self, llm: BaseChatModel):
        super().__init__(
            role="Code Generator",
            goal="Generate high-quality code based on technical specifications",
            backstory="""Expert programmer specialized in implementing technical solutions. 
            You have extensive experience in writing clean, efficient, and maintainable code 
            across multiple programming languages and frameworks.""",
            llm=llm,
            verbose=True
        )
        
    def generate_code(self, tech_spec: TechnicalSpec) -> str:
        """
        Generate code based on technical specifications
        
        Args:
            tech_spec (TechnicalSpec): Technical specifications to implement
            
        Returns:
            str: Generated code implementation
        """
        # The actual code generation will be handled by CrewAI's task system
        # This method is for documentation and type hinting
        pass
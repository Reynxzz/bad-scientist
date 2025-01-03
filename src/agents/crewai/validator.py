from crewai import Agent
from langchain.chat_models.base import BaseChatModel
from agents.base_model import ValidationResult

class ValidatorAgent(Agent):
    """Agent responsible for testing and validating code"""
    
    def __init__(self, llm: BaseChatModel):
        super().__init__(
            role="Code Validator",
            goal="Test and validate generated code",
            backstory="""Quality assurance specialist focused on code testing and validation. 
            You have a keen eye for potential issues and extensive experience in ensuring 
            code quality and reliability.""",
            llm=llm,
            verbose=True
        )
        
    def validate_code(self, code: str) -> ValidationResult:
        """
        Test and validate the generated code
        
        Args:
            code (str): Code to validate
            
        Returns:
            ValidationResult: Validation results and suggestions
        """
        # The actual validation will be handled by CrewAI's task system
        # This method is for documentation and type hinting
        pass
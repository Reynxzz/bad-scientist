"""
LangChain agents implementation module with improved output handling.
"""

from typing import List, Dict, Any
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_mistralai import ChatMistralAI
import re

class AgentPrompts:
    """Contains prompt templates for different agent types."""
    
    @staticmethod
    def get_base_prompt() -> str:
        """Returns the base prompt template structure."""
        return """
        You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Remember to focus on providing clear, concise information without any troubleshooting URLs or unnecessary appendices.

        Begin!

        Question: {input}
        {agent_scratchpad}
        """
    
    @classmethod
    def get_requirements_prompt(cls) -> PromptTemplate:
        """Returns the requirements analysis prompt template."""
        template = f"""You are a requirements analyst. Analyze business requirements and extract key 
        technical components to implement using Python. Tag each requirement with priority and 
        technical scope. Provide your analysis without including any troubleshooting URLs or 
        additional appendices.{cls.get_base_prompt()}"""
        return PromptTemplate.from_template(template)
    
    @classmethod
    def get_sklearn_prompt(cls) -> PromptTemplate:
        """Returns the scikit-learn research prompt template."""
        template = f"""You are a machine learning engineer. Research scikit-learn implementation 
        details based on requirements. Map implementation patterns to requirements. Focus on 
        providing clear technical specifications without any troubleshooting URLs or additional notes.
        {cls.get_base_prompt()}"""
        return PromptTemplate.from_template(template)
    
    @classmethod
    def get_streamlit_prompt(cls) -> PromptTemplate:
        """Returns the Streamlit research prompt template."""
        template = f"""You are a UI developer. Research Streamlit implementation details that 
        integrate with sklearn components. Ensure UI patterns align with business requirements. 
        Provide clear implementation guidance without including any troubleshooting URLs or additional notes. Ouput only python code (using streamlit).
        {cls.get_base_prompt()}"""
        return PromptTemplate.from_template(template)

class LangChainAgent:
    """Base class for LangChain agents with improved output handling."""
    
    def __init__(self, llm: ChatMistralAI, tools: List[Tool], prompt: PromptTemplate):
        self.agent_executor = self._create_agent_executor(llm, tools, prompt)
    
    def _create_agent_executor(self, llm: ChatMistralAI, 
                             tools: List[Tool], 
                             prompt: PromptTemplate) -> AgentExecutor:
        """Create and configure an agent executor."""
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=False  # Disable intermediate steps in output
        )
    
    def _clean_output(self, output: str) -> str:
        """Clean the output by removing unwanted messages and URLs."""
        # Remove troubleshooting URLs and common appendices
        patterns = [
            r'For troubleshooting.*$',
            r'By following these steps.*$',
            r'Parameter.*not yet supported.*$',
            r'https://.*$',
            r'\n\s*$'  # Remove trailing newlines and spaces
        ]
        
        cleaned_output = output
        for pattern in patterns:
            cleaned_output = re.sub(pattern, '', cleaned_output, flags=re.MULTILINE)
        
        return cleaned_output.strip()
    
    def run(self, input_text: str) -> str:
        """Execute the agent with the given input and clean the output."""
        response = self.agent_executor.invoke({"input": input_text})
        output = response['output']
        
        # Clean and format the output
        cleaned_output = self._clean_output(output)
        return cleaned_output
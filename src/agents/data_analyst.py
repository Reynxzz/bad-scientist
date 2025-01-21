from crewai import Agent
from crewai_tools import BaseTool
from typing import List

class DataAnalysisAgent(Agent):
    def __init__(self, llm, tools: List[BaseTool]):
        super().__init__(
            role='Snowflake Data Analyst/Scientist',
            goal='Identify and analyze relevant Snowflake tables that can fulfill the requirements',
            backstory="""You are an expert Snowflake data analyst who understands how to map 
            business requirements to existing data structures. You excel at identifying relevant 
            tables and fields that can support the implementation requirements.""",
            llm=llm,
            tools=tools,
            verbose=True,
            memory=True
        )
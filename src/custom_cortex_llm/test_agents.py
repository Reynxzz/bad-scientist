from snowflake_mistral_agents import CrewSnowflakeLLM 
from snowflake.snowpark.session import Session
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import os
import math
from crewai.tools import BaseTool
from typing import Union, Dict
import math

class CalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = """A statistical calculator that can perform various operations on numbers.
    Input can be either a string in format 'operation: numbers' or a dictionary with 'operation' and 'numbers'.
    Supported operations: mean, sum, std
    Examples: 
    - Calculate mean: 'mean: 1,2,3,4,5' or {'operation': 'mean', 'numbers': '1,2,3,4,5'}
    - Calculate sum: 'sum: 10,20,30' or {'operation': 'sum', 'numbers': '10,20,30'}
    - Calculate standard deviation: 'std: 2,4,4,4,5,5,7,9'"""

    def _parse_input(self, argument: Union[str, Dict]) -> tuple:
        """Parse input whether it's a string or dictionary"""
        if isinstance(argument, dict):
            operation = argument.get('operation', '').strip().lower()
            numbers_str = str(argument.get('numbers', ''))
        else:
            # Handle string input
            try:
                operation, numbers_str = argument.split(':')
                operation = operation.strip().lower()
            except ValueError:
                raise ValueError("String input must be in format 'operation: numbers'")
        
        # Parse numbers
        try:
            numbers = [float(n.strip()) for n in numbers_str.split(',') if n.strip()]
            if not numbers:
                raise ValueError("No valid numbers provided")
        except ValueError as e:
            raise ValueError(f"Error parsing numbers: {str(e)}")
            
        return operation, numbers

    def _run(self, argument: Union[str, Dict]) -> str:
        try:
            # Parse input
            operation, numbers = self._parse_input(argument)
            
            # Perform calculations
            if operation == 'mean':
                result = sum(numbers) / len(numbers)
                return f"The mean is {result:.2f}"
            elif operation == 'sum':
                result = sum(numbers)
                return f"The sum is {result:.2f}"
            elif operation == 'std':
                mean = sum(numbers) / len(numbers)
                squared_diff_sum = sum((x - mean) ** 2 for x in numbers)
                std_dev = math.sqrt(squared_diff_sum / len(numbers))
                return f"The standard deviation is {std_dev:.2f}"
            else:
                return f"Unsupported operation: {operation}. Supported operations are: mean, sum, std"
        except Exception as e:
            return f"Error performing calculation: {str(e)}"

    async def _arun(self, argument: Union[str, Dict]) -> str:
        """Async implementation of the tool"""
        return self._run(argument)
    
def test_crew_agent():
    try:
        # Load environment variables
        load_dotenv()

        # Snowflake connection setup
        connection_params = {
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_USER_PASSWORD"),
            "role": os.getenv("SNOWFLAKE_ROLE"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE")
        }

        # Create Snowflake session
        snowpark_session = Session.builder.configs(connection_params).create()

        # Test the connection
        print("Testing Snowflake connection...")
        test_query = snowpark_session.sql("SELECT 1").collect()
        print("Snowflake connection successful!")

        # Initialize the custom LLM
        print("Initializing LLM...")
        llm = CrewSnowflakeLLM(
            session=snowpark_session,
            model_name="mistral-large2",
            temperature=0.7
        )

        # Create calculator tool instance
        calculator_tool = CalculatorTool()

        # First, test the LLM directly
        print("\nTesting direct LLM call...")
        test_response = llm.call("Say hello!")
        print(f"LLM test response: {test_response}")

        # Create an analyst agent with the calculator tool
        print("\nCreating analyst agent...")
        analyst = Agent(
            role='Data Analyst',
            goal='Analyze data and provide statistical insights',
            backstory="""You are a skilled data analyst who excels at analyzing 
            numerical data and providing clear explanations. You know how to use 
            statistical tools to derive meaningful insights.""",
            allow_delegation=False,
            llm=llm,
            tools=[calculator_tool],
            verbose=True
        )

        # Create a task that will use the calculator
        print("\nCreating analysis task...")
        analysis_task = Task(
            description="""Analyze this sample dataset: 15, 25, 35, 45, 55
            1. Calculate the mean
            2. Calculate the sum
            3. Calculate the standard deviation
            
            Provide a brief explanation of what these numbers tell us about the dataset.""",
            expected_output="""Statistical analysis of the dataset including mean, 
            sum, and standard deviation, with clear explanations of what these 
            metrics indicate about the data.""",
            agent=analyst
        )

        # Create a crew with our analyst
        print("\nCreating analysis crew...")
        analysis_crew = Crew(
            agents=[analyst],
            tasks=[analysis_task],
            process=Process.sequential,
            verbose=True
        )

        # Execute the crew's task and get the result
        print("\nExecuting crew task...")
        result = analysis_crew.kickoff()
        
        print("\n=== Analysis Result ===")
        print(result)
        
        return result

    except Exception as e:
        print(f"\nError during agent test: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test the calculator tool directly first
    print("Testing calculator tool directly:")
    calculator = CalculatorTool()
    
    # Test string inputs
    string_test_cases = [
        "mean: 1,2,3,4,5",
        "sum: 10,20,30",
        "std: 2,4,4,4,5,5,7,9"
    ]
    
    # Test dictionary inputs
    dict_test_cases = [
        {"operation": "mean", "numbers": "1,2,3,4,5"},
        {"operation": "sum", "numbers": "10,20,30"},
        {"operation": "std", "numbers": "2,4,4,4,5,5,7,9"}
    ]
    print("\nTesting string inputs:")
    for test_case in string_test_cases:
        result = calculator._run(test_case)
        print(f"Input: {test_case}")
        print(f"Output: {result}\n")
        
    print("\nTesting dictionary inputs:")
    for test_case in dict_test_cases:
        result = calculator._run(test_case)
        print(f"Input: {test_case}")
        print(f"Output: {result}\n")
    
    print("\nStarting agent test...")
    test_crew_agent()
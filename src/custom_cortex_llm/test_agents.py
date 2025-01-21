from snowflake.snowpark.session import Session
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import os

def test_agent():
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
        print("Creating Snowflake session...")
        session = Session.builder.configs(connection_params).create()

        # Initialize the LLM
        from snowflake_mistral_agents import CrewSnowflakeLLM
        print("\nInitializing LLM...")
        llm = CrewSnowflakeLLM(
            session=session,
            model_name="mistral-large2",
            temperature=0.7
        )

        # Create a simple agent
        print("\nCreating agent...")
        agent = Agent(
            role='Assistant',
            goal='Help answer questions',
            backstory="You are a helpful assistant who provides clear and concise answers.",
            allow_delegation=False,
            llm=llm,
            tools=[],
            verbose=True
        )

        # Create a simple task
        print("\nCreating task...")
        task = Task(
            description="Say hello and introduce yourself briefly.",
            expected_output="A friendly greeting and brief introduction",
            agent=agent,
            tools=[]
        )

        # Create and run crew
        print("\nExecuting task...")
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True  # Changed from 2 to True
        )

        # Execute
        result = crew.kickoff()
        print("\n=== Result ===")
        print(result)
        
        return result

    except Exception as e:
        print(f"\nError during test: {str(e)}")
        raise e

if __name__ == "__main__":
    test_agent()
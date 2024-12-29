from crewai import Agent, Task
from langchain_mistralai import ChatMistralAI

validator_agent = Agent(
    role="Code Validator",
    goal="Test and validate generated Streamlit applications",
    backstory="""You are a thorough QA engineer who ensures all code meets requirements,
    follows best practices, and functions correctly. You catch edge cases and potential
    issues before they become problems.""",
    allow_delegation=True,
    verbose=True,
    tools=[],  # Add testing tools here
    llm=ChatMistralAI(
        model="mistral-large2",
        mistral_api_key="your_mistral_api_key"
    )
)

validate_code_task = Task(
    description="""
    Test and validate the generated application code.
    Ensure all requirements are met and functioning correctly.
    Check for security issues and edge cases.
    Verify AI/ML integrations work as expected.
    """,
    agent=validator_agent
)
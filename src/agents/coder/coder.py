from crewai import Agent, Task
from langchain_mistralai import ChatMistralAI
from base_model import GeneratedCode

coder_agent = Agent(
    role="Code Generator",
    goal="Generate functional Streamlit application code",
    backstory="""You are an expert Python developer specializing in Streamlit applications.
    You write clean, efficient code following best practices and patterns. You ensure all
    code is secure and well-documented.""",
    allow_delegation=True,
    verbose=True,
    tools=[],  # Add code generation tools here
    llm=ChatMistralAI(
        model="mistral-large2",
        mistral_api_key="your_mistral_api_key"
    )
)

generate_code_task = Task(
    description="""
    Generate the complete Streamlit application code based on requirements.
    Implement all required functionality and components.
    Ensure code is clean, efficient, and well-documented.
    """,
    agent=coder_agent,
    output_pydantic=GeneratedCode
)
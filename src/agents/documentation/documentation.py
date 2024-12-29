from crewai import Agent, Task
from langchain_mistralai import ChatMistralAI

documentation_agent = Agent(
    role="Documentation Specialist",
    goal="Research and retrieve relevant technical documentation",
    backstory="""You are a documentation expert who knows how to find and synthesize
    information from Streamlit docs, API references, and internal documentation. You ensure
    all code follows latest best practices.""",
    allow_delegation=True,
    verbose=True,
    tools=[],  # Add your RAG tools here
    llm=ChatMistralAI(
        model="mistral-large2",
        mistral_api_key="your_mistral_api_key"
    )
)

research_documentation_task = Task(
    description="""
    Research relevant documentation for all required components.
    Ensure all implementations follow latest Streamlit best practices.
    Gather information about required AI/ML models and integrations.
    """,
    agent=documentation_agent
)
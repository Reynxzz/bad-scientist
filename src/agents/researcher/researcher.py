from crewai import Agent
from langchain_mistralai import ChatMistralAI

researcher_agent = Agent(
    role="AI/ML Researcher",
    goal="Research and implement AI/ML features",
    backstory="""You are an AI/ML expert who knows how to integrate various models
    and intelligent features into Streamlit apps. You ensure all AI implementations
    are efficient and appropriate for the use case.""",
    allow_delegation=True,
    verbose=True,
    tools=[],  # Add AI/ML tools here
    llm=ChatMistralAI(
        model="mistral-large2",
        mistral_api_key="your_mistral_api_key"
    )
)
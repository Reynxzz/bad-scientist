from crewai import Agent, Task
from langchain_mistralai import ChatMistralAI
from base_model import StreamlitAppRequirements

requirements_agent = Agent(
    role="Requirements Analyzer",
    goal="Analyze and break down app requirements from natural language descriptions",
    backstory="""You are an expert requirements analyst with deep knowledge of Streamlit
    and modern app development practices. You excel at translating business needs into
    technical specifications.""",
    allow_delegation=True,
    verbose=True,
    tools=[],
    llm=ChatMistralAI(
        model="mistral-large2",
        mistral_api_key="your_mistral_api_key"
    )
)

analyze_requirements_task = Task(
    description="""
    Analyze the provided app description and extract detailed requirements.
    Break down the description into specific technical components and functionality needed.
    Consider security implications and data requirements.
    """,
    agent=requirements_agent,
    output_pydantic=StreamlitAppRequirements
)
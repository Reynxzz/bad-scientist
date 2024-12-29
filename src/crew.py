from crewai import Crew, Process, Task
from agents.coder.coder import coder_agent, generate_code_task
from agents.validator.validator import validator_agent, validate_code_task
from agents.requirements.requirements import requirements_agent, analyze_requirements_task
from agents.documentation.documentation import documentation_agent, research_documentation_task
from agents.researcher.researcher import researcher_agent
from agents.base_model import GeneratedCode

# --- Crew ---
crew = Crew(
    agents=[
        requirements_agent,
        documentation_agent,
        coder_agent,
        validator_agent,
        researcher_agent
    ],
    tasks=[
        analyze_requirements_task,
        research_documentation_task,
        generate_code_task,
        validate_code_task
    ],
    process=Process.sequential,
    verbose=2
)

# --- Main Function ---
def generate_streamlit_app(app_description: str) -> GeneratedCode:
    """
    Generate a complete Streamlit application from a text description.
    
    Args:
        app_description (str): Natural language description of the desired app
        
    Returns:
        GeneratedCode: Generated application code and metadata
    """
    result = crew.kickoff(
        inputs={
            "app_description": app_description
        }
    )
    return result

if __name__ == "__main__":
    app_description = """
    Create a Streamlit app that analyzes sentiment from social media posts.
    It should have a text input for users to paste posts, display the sentiment
    score with a gauge chart, and show key phrases that influenced the sentiment.
    """
    
    generated_app = generate_streamlit_app(app_description)
    print(f"Generated app code: {generated_app}")
import streamlit as st
from snowflake.snowpark.session import Session
from snowflake.core import Root
from langchain_mistralai import ChatMistralAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from pathlib import Path
import tempfile
import json
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

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

class LangChainRAGProcessor:
    def __init__(self):
        self.snowpark_session = Session.builder.configs(connection_params).create()
        self.root = Root(self.snowpark_session)
        self.chat_model = ChatMistralAI(
            mistral_api_key=os.environ["MISTRAL_API_KEY"],
            model="mistral-large-latest",
            temperature=0.7
        )
        
    def search_requirements(self, query: str) -> str:
        """Search through requirements documents."""
        service_name = "req_docs_search_svc"
        search_service = (
            self.root
            .databases[self.snowpark_session.get_current_database()]
            .schemas[self.snowpark_session.get_current_schema()]
            .cortex_search_services[service_name]
        )
        
        results = search_service.search(
            query=query,
            columns=["doc_text", "source"],
            limit=5
        )
        
        return json.dumps({
            "results": [
                {"doc_text": r['doc_text'], "source": r['source']}
                for r in results.results
            ]
        })

    def search_technical_docs(self, query: str, tech_stack: str) -> str:
        """Search through technical documentation."""
        service_name = f"{tech_stack}_docs_search_svc"
        search_service = (
            self.root
            .databases[self.snowpark_session.get_current_database()]
            .schemas[self.snowpark_session.get_current_schema()]
            .cortex_search_services[service_name]
        )
        
        results = search_service.search(
            query=query,
            columns=["doc_text"],
            limit=5
        )
        
        return json.dumps({
            "results": [
                {"doc_text": r['doc_text'], "source": tech_stack}
                for r in results.results
            ]
        })

    def setup_requirements_agent(self):
        """Set up agent for requirements analysis."""
        tools = [
            Tool(
                name="SearchRequirements",
                func=self.search_requirements,
                description="Search through business requirements documents"
            )
        ]

        prompt = PromptTemplate.from_template(
            """You are a requirements analyst. Analyze business requirements and extract key technical 
            components to implement using Python. Tag each requirement with priority and technical scope.

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

            Begin!

            Question: {input}
            {agent_scratchpad}"""
        )

        agent = create_react_agent(
            llm=self.chat_model,
            tools=tools,
            prompt=prompt
        )

        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

    def setup_sklearn_agent(self):
        """Set up agent for sklearn research."""
        tools = [
            Tool(
                name="SearchTechnicalDocs",
                func=lambda q: self.search_technical_docs(q, "sklearn"),
                description="Search through scikit-learn documentation"
            )
        ]

        prompt = PromptTemplate.from_template(
            """You are a machine learning engineer. Research scikit-learn implementation details 
            based on requirements. Map implementation patterns to requirements.

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

            Begin!

            Question: {input}
            {agent_scratchpad}"""
        )

        agent = create_react_agent(
            llm=self.chat_model,
            tools=tools,
            prompt=prompt
        )

        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

    def setup_streamlit_agent(self):
        """Set up agent for Streamlit research."""
        tools = [
            Tool(
                name="SearchTechnicalDocs",
                func=lambda q: self.search_technical_docs(q, "streamlit"),
                description="Search through Streamlit documentation"
            )
        ]

        prompt = PromptTemplate.from_template(
            """You are a UI developer. Research Streamlit implementation details that integrate 
            with sklearn components. Ensure UI patterns align with business requirements.

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

            Begin!

            Question: {input}
            {agent_scratchpad}"""
        )

        agent = create_react_agent(
            llm=self.chat_model,
            tools=tools,
            prompt=prompt
        )

        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

    def analyze_requirements(self, prompt: str) -> str:
        """Analyze requirements using LangChain agent."""
        agent = self.setup_requirements_agent()
        response = agent.invoke({"input": f"Analyze these requirements: {prompt}"})
        return response['output']

    def research_sklearn(self, requirements: str) -> str:
        """Research sklearn implementation details."""
        agent = self.setup_sklearn_agent()
        response = agent.invoke({"input": f"Research sklearn implementations for these requirements: {requirements}"})
        return response['output']

    def research_streamlit(self, requirements: str, sklearn_specs: str) -> str:
        """Research Streamlit implementation details."""
        agent = self.setup_streamlit_agent()
        response = agent.invoke({
            "input": f"""Research Streamlit implementations that integrate with these sklearn specs: {sklearn_specs}
            Original requirements: {requirements}"""
        })
        return response['output']

    def generate_code(self, requirements: str, sklearn_specs: str, streamlit_specs: str) -> str:
        """Generate final Python implementation."""
        prompt = PromptTemplate.from_template(
            """You are a Python developer. Generate a complete implementation that integrates 
            Streamlit UI with sklearn components. Include clear documentation linking code to requirements.
            
            Requirements: {requirements}
            Sklearn Specifications: {sklearn_specs}
            Streamlit Specifications: {streamlit_specs}
            """
        )
        
        response = self.chat_model.invoke(prompt.format(
            requirements=requirements,
            sklearn_specs=sklearn_specs,
            streamlit_specs=streamlit_specs
        ))
        
        return response.content

def main():
    st.title("LangChain RAG Application Builder")
    
    # Business idea prompt input
    prompt = st.text_area(
        "Enter your business idea prompt:", 
        help="Describe your business idea or requirements in detail"
    )
    
    # Optional document upload
    uploaded_file = st.file_uploader(
        "Upload business requirements document (optional)",
        type=['pdf', 'txt'],
        help="Upload additional documentation to provide more context"
    )
    
    # Project name input
    project_name = st.text_input(
        "Project Name",
        help="Enter a name for your project"
    )
    
    if st.button("Generate App", type="primary"):
        if prompt and project_name:
            try:
                with st.spinner("Processing your request..."):
                    processor = LangChainRAGProcessor()
                    
                    # Sequential processing with progress updates
                    st.write("Analyzing requirements...")
                    requirements = processor.analyze_requirements(prompt)
                    
                    st.write("Researching scikit-learn implementation...")
                    sklearn_specs = processor.research_sklearn(requirements)
                    
                    st.write("Researching Streamlit implementation...")
                    streamlit_specs = processor.research_streamlit(requirements, sklearn_specs)
                    
                    st.write("Generating final implementation...")
                    final_code = processor.generate_code(requirements, sklearn_specs, streamlit_specs)
                    
                    # Display results
                    st.header("Generated Application")
                    
                    st.subheader("Requirements Analysis")
                    st.markdown(requirements)
                    
                    st.subheader("Scikit-learn Implementation")
                    st.markdown(sklearn_specs)
                    
                    st.subheader("Streamlit Integration")
                    st.markdown(streamlit_specs)
                    
                    st.subheader("Final Implementation")
                    st.code(final_code, language="python")
                    
                    # Save results option
                    if st.button("Save Results"):
                        project_dir = Path(f"projects/{project_name}")
                        project_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save complete results
                        results = {
                            "requirements": requirements,
                            "sklearn_specs": sklearn_specs,
                            "streamlit_specs": streamlit_specs,
                            "implementation": final_code
                        }
                        
                        with open(project_dir / "output.json", "w") as f:
                            json.dump(results, f, indent=2)
                        
                        with open(project_dir / "app.py", "w") as f:
                            f.write(final_code)
                            
                        st.success(f"Results saved to {project_dir}")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()
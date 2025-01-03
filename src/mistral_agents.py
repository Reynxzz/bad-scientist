import streamlit as st
from snowflake.snowpark.session import Session
from snowflake.core import Root
from mistralai import Mistral
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

class MistralRAGProcessor:
    def __init__(self):
        self.snowpark_session = Session.builder.configs(connection_params).create()
        self.root = Root(self.snowpark_session)
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        
    def search_requirements(self, query: str) -> Dict[str, Any]:
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
        
        return {
            "results": [
                {"doc_text": r['doc_text'], "source": r['source']}
                for r in results.results
            ]
        }

    def search_technical_docs(self, query: str, tech_stack: str) -> Dict[str, Any]:
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
        
        return {
            "results": [
                {"doc_text": r['doc_text'], "source": tech_stack}
                for r in results.results
            ]
        }

    def analyze_requirements(self, prompt: str) -> str:
        """Analyze requirements using Mistral."""
        tools = [{
            "type": "function",
            "function": {
                "name": "search_requirements",
                "description": "Search through business requirements documents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to use"
                        }
                    },
                    "required": ["query"]
                }
            }
        }]

        messages = [{
            "role": "system",
            "content": """You are a requirements analyst. Analyze business requirements and extract key technical 
            components to implement using Python. Tag each requirement with priority and technical scope."""
        }, {
            "role": "user",
            "content": f"Analyze these requirements: {prompt}"
        }]

        # Function map for execution
        function_map = {
            "search_requirements": self.search_requirements
        }

        return self._process_mistral_interaction(messages, tools, function_map)

    def research_sklearn(self, requirements: str) -> str:
        """Research sklearn implementation details."""
        tools = [{
            "type": "function",
            "function": {
                "name": "search_technical_docs",
                "description": "Search through technical documentation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to use"
                        },
                        "tech_stack": {
                            "type": "string",
                            "enum": ["sklearn"],
                            "description": "Technology stack to search"
                        }
                    },
                    "required": ["query", "tech_stack"]
                }
            }
        }]

        messages = [{
            "role": "system",
            "content": """You are a machine learning engineer. Research scikit-learn implementation details 
            based on requirements. Map implementation patterns to requirements."""
        }, {
            "role": "user",
            "content": f"Research sklearn implementations for these requirements: {requirements}"
        }]

        # Function map for execution
        function_map = {
            "search_technical_docs": self.search_technical_docs
        }

        return self._process_mistral_interaction(messages, tools, function_map)

    def research_streamlit(self, requirements: str, sklearn_specs: str) -> str:
        """Research Streamlit implementation details."""
        tools = [{
            "type": "function",
            "function": {
                "name": "search_technical_docs",
                "description": "Search through technical documentation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to use"
                        },
                        "tech_stack": {
                            "type": "string",
                            "enum": ["streamlit"],
                            "description": "Technology stack to search"
                        }
                    },
                    "required": ["query", "tech_stack"]
                }
            }
        }]

        messages = [{
            "role": "system",
            "content": """You are a UI developer. Research Streamlit implementation details that integrate 
            with sklearn components. Ensure UI patterns align with business requirements."""
        }, {
            "role": "user",
            "content": f"""Research Streamlit implementations that integrate with these sklearn specs: {sklearn_specs}
            Original requirements: {requirements}"""
        }]

        # Function map for execution
        function_map = {
            "search_technical_docs": self.search_technical_docs
        }

        return self._process_mistral_interaction(messages, tools, function_map)

    def generate_code(self, requirements: str, sklearn_specs: str, streamlit_specs: str) -> str:
        """Generate final Python implementation."""
        messages = [{
            "role": "system",
            "content": """You are a Python developer. Generate a complete implementation that integrates 
            Streamlit UI with sklearn components. Include clear documentation linking code to requirements."""
        }, {
            "role": "user",
            "content": f"""Generate Python implementation based on:
            Requirements: {requirements}
            Sklearn Specifications: {sklearn_specs}
            Streamlit Specifications: {streamlit_specs}"""
        }]

        response = self.client.chat.complete(
            model="mistral-large-latest",
            messages=messages
        )
        
        return response.choices[0].message.content

    def _process_mistral_interaction(self, messages: list, tools: list, function_map: dict) -> str:
        """Process a multi-turn interaction with Mistral including function calling."""
        while True:
            response = self.client.chat.complete(
                model="mistral-large-latest",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            if not response.choices[0].message.tool_calls:
                return response.choices[0].message.content
                
            # Handle tool calls
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Execute function
                function_result = function_map[function_name](**function_args)
                
                # Add interaction to messages
                messages.extend([
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [tool_call]
                    },
                    {
                        "role": "tool",
                        "content": json.dumps(function_result),
                        "tool_call_id": tool_call.id,
                        "name": function_name
                    }
                ])

def main():
    st.title("Mistral RAG Application Builder")
    
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
                    processor = MistralRAGProcessor()
                    
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
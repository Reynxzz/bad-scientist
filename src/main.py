"""
LangChain RAG Application Builder
--------------------------------
Main Streamlit interface for the application.
"""

import streamlit as st
from pathlib import Path
import json
from tools.langchain.rag_processor import RAGProcessor
from tools.langchain.project_manager import ProjectManager

def main():
    """Main application entry point."""
    st.title("LangChain RAG Application Builder")
    
    # Initialize session state for storing generated code
    if 'generated_code' not in st.session_state:
        st.session_state.generated_code = None
    if 'app_results' not in st.session_state:
        st.session_state.app_results = None
    
    # Input collection
    prompt = st.text_area(
        "Enter your business idea prompt:", 
        help="Describe your business idea or requirements in detail"
    )
    
    uploaded_file = st.file_uploader(
        "Upload business requirements document (optional)",
        type=['pdf', 'txt'],
        help="Upload additional documentation to provide more context"
    )
    
    project_name = st.text_input(
        "Project Name",
        help="Enter a name for your project"
    )
    
    if st.button("Generate App", type="primary"):
        if not (prompt and project_name):
            st.error("Please provide both a prompt and project name.")
            return
            
        try:
            with st.spinner("Processing your request..."):
                # Process requirements
                processor = RAGProcessor()
                
                # Show progress
                progress_text = st.empty()
                
                progress_text.write("Analyzing requirements...")
                results = processor.process_requirements(prompt)
                
                # Store results in session state
                st.session_state.generated_code = results["implementation"]
                st.session_state.app_results = results
                
                # Display results
                st.header("Generated Application")
                
                st.subheader("Requirements Analysis")
                st.markdown(results["requirements"])
                
                st.subheader("Scikit-learn Implementation")
                st.markdown(results["sklearn_specs"])
                
                st.subheader("Streamlit Integration")
                st.markdown(results["streamlit_specs"])
                
                st.subheader("Final Implementation")
                st.code(results["implementation"], language="python")
                
                # Save results option
                if st.button("Save Results"):
                    project_dir = ProjectManager.save_project(project_name, results)
                    st.success(f"Results saved to {project_dir}")
                
                # Navigate to generated app
                st.success("App generated successfully! Go to the 'Generated App' page to view it.")
                    
        except Exception as e:
            st.error("An error occurred during processing")
            st.exception(e)

if __name__ == "__main__":
    main()
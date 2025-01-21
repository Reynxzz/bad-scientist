from pathlib import Path
import streamlit as st
import tempfile
import os
from snowflake.snowpark.session import Session
from config import CONNECTION_PARAMETER
from agents.custom_agents import DataScienceFlow
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Bad Scientist",
    page_icon="🧪",
)

def init_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.is_generating = False
        st.session_state.generated_code = None
        st.session_state.app_results = None
        st.session_state.current_progress = 0
        st.session_state.status_message = ""
        st.session_state.debug_messages = []
        st.session_state.snowflake_session = None

def init_snowflake_session():
    """Initialize Snowflake session if not already present"""
    if not st.session_state.snowflake_session:
        try:
            st.session_state.snowflake_session = Session.builder.configs(CONNECTION_PARAMETER).create()
            logger.info("Snowflake session initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Snowflake session: {str(e)}")
            st.error("Failed to connect to Snowflake. Please check your credentials.")
            return False
    return True

def update_progress(status_text, progress_bar, message, progress):
    """Update progress indicators"""
    status_text.write(message)
    progress_bar.progress(progress)
    st.session_state.current_progress = progress
    st.session_state.status_message = message
    logger.debug(f"Progress update: {progress}% - {message}")
    st.session_state.debug_messages.append(f"Progress: {progress}% - {message}")

def execute_flow(prompt: str, status_text, progress_bar):
    """Execute DataScienceFlow with progress updates"""
    try:
        flow = DataScienceFlow(session=st.session_state.snowflake_session)
        
        update_progress(status_text, progress_bar, "Analyzing data requirements...", 25)
        result = flow.execute_flow(prompt)
        
        update_progress(status_text, progress_bar, "Generating machine learning code...", 50)
        
        update_progress(status_text, progress_bar, "Creating visualizations...", 75)
        
        update_progress(status_text, progress_bar, "Finalizing Streamlit application...", 90)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in flow execution: {str(e)}")
        raise

def main():
    init_session_state()
    st.title("Bad Scientist")
    st.write("Text-to-Streamlit App Builder with Snowflake Data Warehouse Integration")
    
    if not st.session_state.is_generating:
        prompt = st.text_area(
            "Enter your business idea prompt:", 
            help="Describe your data analysis requirements and visualization needs"
        )
        
        uploaded_file = st.file_uploader(
            "Upload business requirements document (e.g. RFC/PRD)",
            type=['pdf', 'txt'],
            help="Upload additional documentation to provide more context"
        )
        
        if st.button("Generate App", type="primary", key="generate_initial"):
            if prompt:
                if not init_snowflake_session():
                    return
                
                try:
                    st.session_state.is_generating = True
                    st.session_state.debug_messages = []
                    
                    with st.spinner("Generating app..."):
                        # Create status containers
                        status_text = st.empty()
                        progress_bar = st.progress(0)
                        
                        # Initial status
                        update_progress(status_text, progress_bar, "Starting app generation...", 0)
                        
                        # Execute flow
                        result = execute_flow(prompt, status_text, progress_bar)
                        
                        if not result:
                            raise ValueError("Flow execution failed - no result returned")
                        
                        # Store results in session state
                        st.session_state.generated_code = result.streamlit_code
                        st.session_state.app_results = {
                            "requirements": "Data Science Flow Requirements:\n" + prompt,
                            "data_analysis": result.data_analysis,
                            "reference_patterns": {
                                "Machine Learning": result.sklearn_code,
                                "Visualization": result.visualization_code
                            },
                            "streamlit_components": result.streamlit_code
                        }
                        
                        # Clear status indicators
                        status_text.empty()
                        progress_bar.empty()
                        
                        st.success("Generation complete! Navigate to the Generated App page to view and run your application.")
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)
                    logger.exception("Error during app generation")
                    st.session_state.is_generating = False
                finally:
                    if st.button("Generate Another App", key="generate_another"):
                        st.session_state.is_generating = False
                        st.rerun()
            else:
                st.error("Please enter a business idea prompt")
    else:
        st.success("Generation is complete! Navigate to the **Generated App** page to view and run your application.")
        if st.button("Generate Another App", key="generate_another_main"):
            st.session_state.is_generating = False
            st.rerun()

if __name__ == "__main__":
    main()
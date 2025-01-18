from pathlib import Path
import streamlit as st
import tempfile
import os
import asyncio
from agents.async_crew import execute_streamlit_generation
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

def update_progress(status_text, progress_bar, message, progress):
    """Update progress indicators"""
    status_text.write(message)
    progress_bar.progress(progress)
    st.session_state.current_progress = progress
    st.session_state.status_message = message
    logger.debug(f"Progress update: {progress}% - {message}")
    st.session_state.debug_messages.append(f"Progress: {progress}% - {message}")

def safe_get_attr(obj, attr, default=None):
    """Safely get attribute from object with default value"""
    return getattr(obj, attr, default) if obj else default

def main():
    init_session_state()
    st.title("Bad Scientist")
    st.write("Text-to-Streamlit App Builder with Snowflake Data Warehouse Integration")
    
    if not st.session_state.is_generating:
        prompt = st.text_area(
            "Enter your business idea prompt:", 
            help="Describe your business idea or requirements in detail"
        )
        
        uploaded_file = st.file_uploader(
            "Upload business requirements document (e.g. RFC/PRD)",
            type=['pdf', 'txt'],
            help="Upload additional documentation to provide more context"
        )
        
        if st.button("Generate App", type="primary"):
            if prompt:
                try:
                    st.session_state.is_generating = True
                    st.session_state.debug_messages = []  # Reset debug messages
                    docs_path = None
                    docs_uploaded = False
                    
                    if uploaded_file:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            docs_path = tmp_file.name
                            docs_uploaded = True
                            logger.debug(f"Uploaded file saved to: {docs_path}")
                    
                    with st.spinner("Generating app..."):
                        try:
                            # Create status containers
                            status_text = st.empty()
                            progress_bar = st.progress(0)
                            
                            # Initial status
                            update_progress(status_text, progress_bar, "Starting app generation...", 0)
                            
                            # Execute async flow
                            flow_result = asyncio.run(execute_streamlit_generation(
                                prompt=prompt,
                                docs_uploaded=docs_uploaded,
                                docs_path=docs_path
                            ))
                            
                            if not flow_result:
                                raise ValueError("Flow execution failed - no result returned")
                            
                            logger.debug(f"Flow result: {flow_result}")
                            st.session_state.debug_messages.append(f"Flow result type: {type(flow_result)}")
                            
                            # Store results in session state
                            st.session_state.generated_code = flow_result.final_code
                            st.session_state.app_results = {
                                "requirements": flow_result.requirements,
                                "data_analysis": flow_result.data_analysis,
                                "reference_patterns": flow_result.reference_patterns,
                                "streamlit_components": flow_result.streamlit_components
                            }
                            
                            if not st.session_state.generated_code:
                                raise ValueError("No code was generated")
                            
                            # Clear status indicators
                            status_text.empty()
                            progress_bar.empty()
                            
                            st.success("Generation complete! Navigate to the Generated App page to view and run your application.")
                            if st.button("Generate Another App"):
                                st.session_state.is_generating = False
                                st.rerun()       
                                                     
                        except Exception as e:
                            logger.error(f"Error during flow execution: {str(e)}")
                            raise
                            
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)
                    logger.exception("Error during app generation")
                    st.session_state.is_generating = False
                finally:
                    # Cleanup
                    if docs_path:
                        try:
                            os.unlink(docs_path)
                            logger.debug(f"Cleaned up temporary file: {docs_path}")
                        except Exception as e:
                            logger.warning(f"Could not remove temporary file: {str(e)}")
                            st.warning(f"Could not remove temporary file: {str(e)}")
                    
                    if st.button("Generate Another App"):
                        st.session_state.is_generating = False
                        st.rerun()
            else:
                st.error("Please enter a business idea prompt")
    else:
        st.success("Generation is complete! Navigate to the **Generated App** page to view and run your application.")
        if st.button("Generate Another App"):
            st.session_state.is_generating = False
            st.rerun()

if __name__ == "__main__":
    main()
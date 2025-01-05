from pathlib import Path
import streamlit as st
import tempfile
import os
from agents.crew import create_crew

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
                    with st.spinner("Initializing generation process..."):
                        docs_path = None
                        docs_uploaded = False
                        if uploaded_file:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                docs_path = tmp_file.name
                                docs_uploaded = True
                        
                        crew = create_crew(prompt, docs_uploaded, docs_path)
                        
                        st.write("Starting generation process...")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.write("Analyzing requirements...")
                        crew.kickoff()
                        progress_bar.progress(25)
                        
                        requirement_task = crew.tasks[0]
                        data_analysis_task = crew.tasks[1]
                        sklearn_task = crew.tasks[2]
                        streamlit_task = crew.tasks[3]
                        coder_task = crew.tasks[4]
                        
                        status_text.write("Processing scikit-learn implementation...")
                        progress_bar.progress(50)
                        
                        status_text.write("Generating Streamlit interface...")
                        progress_bar.progress(75)
                        
                        status_text.write("Creating final implementation...")
                        progress_bar.progress(100)
                        
                        # Store results in session state
                        st.session_state.generated_code = coder_task.output.raw
                        st.session_state.app_results = {
                            "requirements": requirement_task.output.raw,
                            "data_needed": data_analysis_task.output.raw,
                            "sklearn_specs": sklearn_task.output.raw,
                            "streamlit_specs": streamlit_task.output.raw,
                        }
                        
                        st.success("Generation complete! Navigate to the Generated App page to view and run your application.")
                        
                        if st.button("Generate Another App"):
                            st.session_state.is_generating = False
                            st.experimental_rerun()
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)
                    st.session_state.is_generating = False
                finally:
                    # Cleanup
                    if docs_path:
                        try:
                            os.unlink(docs_path)
                        except Exception as e:
                            st.warning(f"Could not remove temporary file: {str(e)}")
            else:
                st.error("Please enter a business idea prompt")
    else:
        st.success("Generation is complete! Navigate to the **Generated App** page to view and run your application.")
        if st.button("Generate Another App"):
            st.session_state.is_generating = False
            st.rerun()

if __name__ == "__main__":
    main()
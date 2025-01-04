from pathlib import Path
import streamlit as st
import tempfile
import os
from agents.crew import create_crew

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
    # Initialize session state
    init_session_state()
    
    st.title("Bad Scientist: Streamlit App Builder")
    
    # Only show inputs if not currently generating
    if not st.session_state.is_generating:
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
        
        if st.button("Generate App", type="primary"):
            if prompt:
                try:
                    st.session_state.is_generating = True
                    with st.spinner("Initializing generation process..."):
                        # Handle document upload
                        docs_path = None
                        if uploaded_file:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                docs_path = tmp_file.name
                        
                        # Create crew
                        crew = create_crew(prompt, docs_path)
                        
                        # Progress indicators
                        st.write("Starting generation process...")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Execute tasks
                        status_text.write("Analyzing requirements...")
                        crew.kickoff()
                        progress_bar.progress(25)
                        
                        # Get individual task results
                        requirement_task = crew.tasks[0]
                        sklearn_task = crew.tasks[1]
                        streamlit_task = crew.tasks[2]
                        coder_task = crew.tasks[3]
                        
                        # Update progress for each task
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
                            "sklearn_specs": sklearn_task.output.raw,
                            "streamlit_specs": streamlit_task.output.raw,
                        }
                        
                        # Success message with debug info
                        st.success("Generation complete! Navigate to the Generated App page to view and run your application.")
                        
                        # Add a reset button
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
        # Show status when generation is complete
        st.success("Generation is complete! Navigate to the Generated App page to view and run your application.")
        if st.button("Generate Another App"):
            st.session_state.is_generating = False
            st.rerun()

if __name__ == "__main__":
    main()
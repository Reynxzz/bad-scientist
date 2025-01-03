"""
Generated Application Display
---------------------------
Displays and runs the generated Streamlit application.
"""

import streamlit as st
import sys
from pathlib import Path
import importlib.util

def load_and_run_generated_code(code_string: str):
    """
    Safely loads and runs the generated code in the current Streamlit context.
    
    Args:
        code_string: String containing the Python code to execute
    """
    try:
        # Create a temporary module
        spec = importlib.util.spec_from_loader(
            "generated_module", 
            loader=None
        )
        module = importlib.util.module_from_spec(spec)
        
        # Add streamlit to the module's namespace
        module.st = st
        
        # Execute the code in the module's namespace
        exec(code_string, module.__dict__)
        
        # If the code has a main() function, run it
        if hasattr(module, 'main'):
            module.main()
            
    except Exception as e:
        st.error("Error running generated application")
        st.exception(e)

def main():
    st.title("Generated Application")
    
    if st.session_state.get('generated_code') is None:
        st.warning("No application has been generated yet. Please go to the home page to generate an application.")
        return
    
    # Display tabs for viewing code and running app
    tab1, tab2 = st.tabs(["Run Application", "View Code"])
    
    with tab1:
        st.header("Running Application")
        if st.session_state.generated_code:
            load_and_run_generated_code(st.session_state.app_results["streamlit_specs"])
    
    with tab2:
        st.header("Application Code")
        if st.session_state.generated_code:
            st.code(st.session_state.generated_code, language="python")
            
            # Display other artifacts if available
            if st.session_state.get('app_results'):
                with st.expander("View Requirements Analysis"):
                    st.markdown(st.session_state.app_results["requirements"])
                
                with st.expander("View Scikit-learn Implementation Details"):
                    st.markdown(st.session_state.app_results["sklearn_specs"])
                
                with st.expander("View Streamlit Integration Details"):
                    st.markdown(st.session_state.app_results["streamlit_specs"])

if __name__ == "__main__":
    main()
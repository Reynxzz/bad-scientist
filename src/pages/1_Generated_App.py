import streamlit as st
import sys
from pathlib import Path
import importlib.util

def load_and_run_generated_code(code_string: str):
    """
    Safely loads and runs the generated codes in the current Streamlit context.
    """
    try:
        code_string = code_string.replace('```python', '').replace('```', '').strip()
        spec = importlib.util.spec_from_loader(
            "generated_module", 
            loader=None
        )
        module = importlib.util.module_from_spec(spec)        
        module.st = st
        
        # Execute the code in the module's namespace
        exec(code_string, module.__dict__)
        
        if hasattr(module, 'main'):
            module.main()
            
    except Exception as e:
        st.error("Error running generated application")
        st.exception(e)

def main():
    st.title("Generated Application")
    
    if st.session_state.get('generated_code') is None:
        st.warning("No application has been generated yet. Please go to the Generator page to create an application.")
        return
    
    tab1, tab2 = st.tabs(["Run Application", "View Details & Code"])
    
    with tab1:
        st.write("Running Generated Application...")
        if st.session_state.generated_code:
            load_and_run_generated_code(st.session_state.generated_code)
    
    with tab2:
        if st.session_state.get('app_results'):
            # Requirements Analysis
            with st.expander("Requirements Analysis", expanded=True):
                st.subheader("Analysis Description")
                st.markdown(st.session_state.app_results["requirements"])
            
            # Scikit-learn Implementation
            with st.expander("Scikit-learn Implementation", expanded=False):
                st.subheader("Implementation Approach")
                st.markdown(st.session_state.app_results["sklearn_specs"])
            
            # Streamlit Integration
            with st.expander("Streamlit Integration", expanded=False):
                st.subheader("Integration Approach")
                st.markdown(st.session_state.app_results["streamlit_specs"])
            
            # Final Implementation
            with st.expander("Final Implementation", expanded=False):
                st.subheader("Generated Code")
                st.code(st.session_state.generated_code, language="python")

if __name__ == "__main__":
    main()
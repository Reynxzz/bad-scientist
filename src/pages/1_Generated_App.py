import streamlit as st
import re
import importlib.util

st.set_page_config(
    page_title="Generated Apps",
    page_icon="🧪"
)

def extract_python_code(text: str) -> str:
    """
    Extracts Python code from markdown code blocks.
    Handles variations in fence styles and whitespace.
    
    Args:
        text (str): Text containing markdown code blocks
        
    Returns:
        str: Extracted Python code with proper whitespace
    """
    # Pattern matches:
    # 1. ```python or ``` python (optional whitespace)
    # 2. Code content (non-greedy match)
    # 3. Closing ``` fence
    pattern = r'```\s*python\s*(.*?)\s*```'
    
    # Find all matches using re.DOTALL to match across lines
    matches = re.findall(pattern, text, re.DOTALL)
    
    if not matches:
        return text.strip()
        
    return matches[0].strip()

def load_and_run_generated_code(code_string: str):
    """
    Safely loads and runs the generated codes in the current Streamlit context.
    """
    try:
        code_string = extract_python_code(code_string)
        code_string = code_string.replace('st.set_page_config(', '#')
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
            
            with st.expander("Data Analysis", expanded=True):
                st.subheader("Data Needed")
                st.markdown(st.session_state.app_results["data_needed"])
            
            # Scikit-learn Implementation
            with st.expander("Streamlit App Implementation Based on Reference", expanded=False):
                st.subheader("Implementation Approach")
                st.markdown(st.session_state.app_results["st_ref"])
            
            # Streamlit Integration
            with st.expander("Streamlit Integration", expanded=False):
                st.subheader("Integration Approach")
                st.markdown(st.session_state.app_results["streamlit_specs"])
            
            # Final Implementation
            with st.expander("Final Implementation", expanded=False):
                st.subheader("Generated Code")
                st.markdown(st.session_state.generated_code)

if __name__ == "__main__":
    main()
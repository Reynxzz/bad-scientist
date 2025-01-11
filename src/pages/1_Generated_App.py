# pages/1_Generated_App.py
import streamlit as st
import re
import importlib.util

def init_session_state():
    """Initialize session state variables for the generated app"""
    if 'generated_code' not in st.session_state:
        st.session_state.generated_code = None
    if 'app_results' not in st.session_state:
        st.session_state.app_results = None
    if 'app_state' not in st.session_state:
        st.session_state.app_state = {}

def extract_python_code(text: str) -> str:
    """Extracts Python code from markdown code blocks or raw text."""
    if not text:
        return ""
    
    # First try to extract from markdown code blocks
    pattern = r'```\s*python\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # If no markdown blocks found, check for ```code``` without language
    pattern = r'```\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # If no code blocks found, treat entire text as code
    return text.strip()

def sanitize_code(code_string: str) -> str:
    """Sanitizes the code string for safe execution."""
    # Remove any page config calls
    code_string = re.sub(
        r'st\.set_page_config\([^)]*\)',
        '# Page config removed',
        code_string
    )
    
    return code_string.strip()

def format_code_for_display(code: str) -> str:
    """Formats code for display in markdown."""
    return f"```python\n{code}\n```"

class StateManager:
    """Manages state persistence for the generated application."""
    def __init__(self):
        self.states = {}
    
    def get_state(self, key, default=None):
        return self.states.get(key, default)
    
    def set_state(self, key, value):
        self.states[key] = value

def create_wrapped_streamlit():
    """Creates a wrapped version of streamlit that preserves widget states."""
    state_manager = StateManager()
    
    class WrappedStreamlit:
        @staticmethod
        def number_input(*args, **kwargs):
            key = kwargs.get('key', str(args))
            if 'value' not in kwargs:
                kwargs['value'] = state_manager.get_state(key, 0.0)
            value = st.number_input(*args, **kwargs)
            state_manager.set_state(key, value)
            return value

        @staticmethod
        def text_input(*args, **kwargs):
            key = kwargs.get('key', str(args))
            if 'value' not in kwargs:
                kwargs['value'] = state_manager.get_state(key, '')
            value = st.text_input(*args, **kwargs)
            state_manager.set_state(key, value)
            return value

        @staticmethod
        def selectbox(*args, **kwargs):
            key = kwargs.get('key', str(args))
            if 'index' not in kwargs:
                kwargs['index'] = state_manager.get_state(key, 0)
            value = st.selectbox(*args, **kwargs)
            state_manager.set_state(key, args[1].index(value) if args[1] else 0)
            return value

        def __getattr__(self, name):
            # Pass through any other Streamlit functions unchanged
            return getattr(st, name)

    return WrappedStreamlit()

def load_generated_code(code_string: str):
    """Safely loads and executes the generated code with state preservation."""
    try:
        # Extract and sanitize code
        code_string = extract_python_code(code_string)
        code_string = sanitize_code(code_string)
        
        if not code_string:
            st.error("No valid code found to execute")
            return
        
        # Create module for code execution
        spec = importlib.util.spec_from_loader(
            "generated_module", 
            loader=None
        )
        module = importlib.util.module_from_spec(spec)
        
        # Create wrapped streamlit instance
        wrapped_st = create_wrapped_streamlit()
        module.st = wrapped_st
        
        # Execute the code
        exec(code_string, module.__dict__)
        
        # Run main function if it exists
        if hasattr(module, 'main'):
            module.main()
            
    except Exception as e:
        st.error("Error running generated application")
        with st.expander("Error Details", expanded=True):
            st.exception(e)
            st.code(code_string, language="python")

def display_app_details():
    """Displays the application details and code."""
    if st.session_state.get('app_results'):
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("Technical Requirements", expanded=True):
                if requirements := st.session_state.app_results.get("requirements"):
                    st.markdown(requirements)
                else:
                    st.info("No requirements analysis available")
            
            with st.expander("Data Analysis", expanded=True):
                if data_analysis := st.session_state.app_results.get("data_analysis"):
                    st.markdown(data_analysis)
                else:
                    st.info("No data analysis available")
        
        with col2:
            with st.expander("Implementation Patterns", expanded=True):
                if patterns := st.session_state.app_results.get("reference_patterns"):
                    if isinstance(patterns, dict):
                        for pattern_type, pattern in patterns.items():
                            st.markdown(f"### {pattern_type.title()}")
                            st.markdown(pattern)
                    else:
                        st.markdown(patterns)
                else:
                    st.info("No implementation patterns available")
            
            with st.expander("Streamlit Components", expanded=True):
                if components := st.session_state.app_results.get("streamlit_components"):
                    st.markdown(components)
                else:
                    st.info("No component analysis available")
        
        with st.expander("Generated Code", expanded=False):
            if code := st.session_state.generated_code:
                if st.button("Copy Code"):
                    st.code(code, language="python")
                else:
                    st.markdown(format_code_for_display(code))
            else:
                st.warning("No code has been generated")

def main():
    init_session_state()
    
    if st.session_state.get('generated_code') is None:
        st.warning("No application has been generated yet. Please go to the Generator page to create an application.")
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["Application", "Details & Code"])
    
    with tab1:
        # Load and display the application
        if st.session_state.generated_code:
            load_generated_code(st.session_state.generated_code)
    
    with tab2:
        display_app_details()

if __name__ == "__main__":
    main()
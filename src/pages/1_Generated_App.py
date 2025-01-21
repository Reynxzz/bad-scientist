import streamlit as st
import re
from streamlit_ace import st_ace
import matplotlib.pyplot as plt
import pandas as pd
from snowflake.snowpark.session import Session
from config import CONNECTION_PARAMETER

class WrappedStreamlit:
    """Wrapper for Streamlit to preserve widget states"""
    def __init__(self):
        self.widget_states = {}
    
    def _get_widget_key(self, name, args, kwargs):
        """Generate a consistent key for widgets"""
        key = kwargs.get('key', f"{name}_{str(args)}")
        return str(key)
    
    def __getattr__(self, name):
        """Handle Streamlit function calls"""
        attr = getattr(st, name)
        if callable(attr) and name in [
            'slider', 'selectbox', 'multiselect', 'button',
            'checkbox', 'radio', 'number_input', 'text_input',
            'text_area', 'date_input', 'time_input'
        ]:
            def wrapped(*args, **kwargs):
                widget_key = self._get_widget_key(name, args, kwargs)
                
                # Preserve previous value if it exists
                if widget_key in self.widget_states:
                    if 'value' not in kwargs:
                        kwargs['value'] = self.widget_states[widget_key]
                
                # Call original Streamlit function
                result = attr(*args, **kwargs)
                
                # Store new value
                self.widget_states[widget_key] = result
                return result
            return wrapped
        return attr

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'generated_code': None,
        'app_results': None,
        'edited_code': None,
        'previous_code': None,
        'editor_theme': 'terminal',
        'editor_keybinding': 'vscode',
        'editor_font_size': 14,
        'editor_tab_size': 4,
        'snowflake_session': None,
        'is_app_running': False,
        'wrapped_st': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def get_snowflake_session():
    """Get or create Snowflake session"""
    if st.session_state.snowflake_session is None:
        try:
            st.session_state.snowflake_session = Session.builder.configs(CONNECTION_PARAMETER).create()
        except Exception as e:
            st.error(f"Failed to connect to Snowflake: {str(e)}")
            return None
    return st.session_state.snowflake_session

def extract_python_code(text: str) -> str:
    """Extract Python code from text"""
    if not text:
        return ""
    
    patterns = [
        r'```\s*python\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
    
    return text.strip()

def execute_code(code: str):
    """Execute the code with preserved widget states"""
    try:
        # Initialize wrapped Streamlit if not exists
        if st.session_state.wrapped_st is None:
            st.session_state.wrapped_st = WrappedStreamlit()
        
        # Create execution namespace
        namespace = {
            'st': st.session_state.wrapped_st,
            'plt': plt,
            'pd': pd,
            'get_snowflake_session': get_snowflake_session,
            '__name__': '__main__'
        }
        
        # Execute the code
        exec(code, namespace)
        
        # Call main() if it exists
        if 'main' in namespace:
            namespace['main']()
        
        st.session_state.is_app_running = True
            
    except Exception as e:
        st.error("Error executing code:")
        st.exception(e)

def display_app_details():
    """Display application details in expanders"""
    if st.session_state.get('app_results'):
        with st.expander("Technical Requirements", expanded=False):
            if requirements := st.session_state.app_results.get("requirements"):
                st.markdown(requirements)
            else:
                st.info("No requirements analysis available")
        
        with st.expander("Data Analysis", expanded=False):
            if data_analysis := st.session_state.app_results.get("data_analysis"):
                st.markdown(data_analysis)
            else:
                st.info("No data analysis available")
        
        with st.expander("Further Implementation", expanded=False):
            if patterns := st.session_state.app_results.get("reference_patterns"):
                if isinstance(patterns, dict):
                    # Only show Machine Learning code
                    if ml_code := patterns.get("Machine Learning"):
                        st.code(ml_code, language="python")
                else:
                    st.info("No implementation patterns available")
            else:
                st.info("No implementation patterns available")

def main():
    init_session_state()
    
    if st.session_state.generated_code is None:
        st.warning("No application has been generated yet. Please go to the Generator page to create an application.")
        return
    
    tab1, tab2 = st.tabs(["Application", "Details"])
    
    with tab1:
        # Code editor section
        if st.session_state.edited_code is None:
            st.session_state.edited_code = extract_python_code(st.session_state.generated_code)

        # Editor and run button
        st.session_state.edited_code = st_ace(
            value=st.session_state.edited_code,
            language="python",
            theme=st.session_state.editor_theme,
            key="ace_editor",
            height=500,
            font_size=st.session_state.editor_font_size,
            tab_size=st.session_state.editor_tab_size,
            wrap=True,
            auto_update=True
        )

        # Run button container
        run_col1, run_col2 = st.columns([6, 1])
        with run_col1:
            if st.button("▶️ Run Code", type="primary", use_container_width=True):
                st.session_state.is_app_running = True
                with st.spinner("Running code..."):
                    st.markdown("### Output:")
                    execute_code(st.session_state.edited_code)
        
        with run_col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.wrapped_st = None
                st.session_state.is_app_running = False
                st.rerun()

        # Execute code if app is running
        if st.session_state.is_app_running:
            st.markdown("### Output:")
            execute_code(st.session_state.edited_code)
    
    with tab2:
        display_app_details()

if __name__ == "__main__":
    main()
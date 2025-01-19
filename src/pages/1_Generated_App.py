import streamlit as st
import re
from streamlit_ace import st_ace

st.set_page_config(
    page_title="Generated App by Bad Scientist",
    page_icon="🧪",
)

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'generated_code': None,
        'app_results': None,
        'edited_code': None,
        'editor_theme': 'terminal',
        'editor_keybinding': 'vscode',
        'editor_font_size': 14,
        'editor_tab_size': 4,
        'widget_states': {},
        'is_first_run': True,
        'is_executing': False,
        'execution_requested': False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def extract_python_code(text: str) -> str:
    """Extracts Python code from markdown code blocks or raw text."""
    if not text:
        return ""
    
    pattern = r'```\s*python\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    pattern = r'```\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    return text.strip()

def sanitize_code(code_string: str) -> str:
    """Sanitizes the code string for safe execution."""
    code_string = re.sub(
        r'st\.set_page_config\([^)]*\)',
        '# Page config removed',
        code_string
    )
    return code_string.strip()

class WrappedStreamlit:
    """A wrapper for Streamlit that preserves widget states."""
    
    def __init__(self):
        if 'widget_states' not in st.session_state:
            st.session_state.widget_states = {}
        self.widget_states = st.session_state.widget_states
        
    def _get_widget_id(self, name, args, kwargs):
        """Generate a unique ID for a widget."""
        key = kwargs.get('key', f"{name}_{str(args)}")
        return str(key)
    
    def _wrap_widget(self, name, *args, **kwargs):
        """Wrap a Streamlit widget to preserve its state."""
        widget_id = self._get_widget_id(name, args, kwargs)
        
        original_widget = getattr(st, name)
        
        if widget_id in self.widget_states:
            stored_value = self.widget_states[widget_id]
            if 'value' not in kwargs and 'default_value' not in kwargs:
                kwargs['value'] = stored_value
        
        try:
            result = original_widget(*args, **kwargs)
            self.widget_states[widget_id] = result
            return result
        except Exception as e:
            if widget_id in self.widget_states:
                return self.widget_states[widget_id]
            raise e

    def __getattr__(self, name):
        """Handle all Streamlit function calls."""
        original_attr = getattr(st, name)
        
        if callable(original_attr):
            def wrapped(*args, **kwargs):
                if name in ['button', 'checkbox', 'number_input', 'text_input', 
                           'selectbox', 'multiselect', 'slider', 'select_slider',
                           'date_input', 'time_input', 'text_area', 'radio']:
                    return self._wrap_widget(name, *args, **kwargs)
                return original_attr(*args, **kwargs)
            return wrapped
        return original_attr

def execute_generated_code():
    """Execute the generated/edited code while preserving widget states."""
    if not st.session_state.edited_code or not st.session_state.execution_requested:
        return

    if st.session_state.is_executing:
        return
        
    try:
        st.session_state.is_executing = True
        
        code = sanitize_code(extract_python_code(st.session_state.edited_code))
        if not code:
            st.error("No valid code found to execute")
            return
        
        wrapped_st = WrappedStreamlit()
        namespace = {
            'st': wrapped_st,
            '__name__': '__main__'
        }
        
        exec(code, namespace)
        
        if 'main' in namespace:
            namespace['main']()
            
    except Exception as e:
        st.error("Error running generated application")
        st.exception(e)
    finally:
        st.session_state.is_executing = False
        st.session_state.execution_requested = False

def display_app_details():
    """Displays the application details in expandable sections."""
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
        
        with st.expander("Implementation Reference/Patterns", expanded=False):
            if patterns := st.session_state.app_results.get("reference_patterns"):
                if isinstance(patterns, dict):
                    for pattern_type, pattern in patterns.items():
                        st.markdown(f"### {pattern_type.title()}")
                        st.markdown(pattern)
                else:
                    st.markdown(patterns)
            else:
                st.info("No implementation patterns available")
        
        with st.expander("Streamlit Components", expanded=False):
            if components := st.session_state.app_results.get("streamlit_components"):
                st.markdown(components)
            else:
                st.info("No component analysis available")

def code_editor_interface():
    """Provides the code editor interface."""
    with st.form("code_editor_form"):
        if st.session_state.edited_code is None and st.session_state.generated_code is not None:
            st.session_state.edited_code = extract_python_code(st.session_state.generated_code)

        # Code editor
        st.session_state.edited_code = st_ace(
            value=st.session_state.edited_code,
            language="python",
            theme=st.session_state.editor_theme,
            key="ace_editor",
            height=600,
            font_size=st.session_state.editor_font_size,
            tab_size=st.session_state.editor_tab_size,
            wrap=True,
            auto_update=True,
            readonly=False,
            min_lines=20,
            keybinding=st.session_state.editor_keybinding
        )

        # Run button inside form
        if st.form_submit_button("Run Code", use_container_width=True, type="primary"):
            st.session_state.execution_requested = True
            st.rerun()

def main():
    init_session_state()
    
    if st.session_state.generated_code is None:
        st.warning("No application has been generated yet. Please go to the Generator page to create an application.")
        return
    
    tab1, tab2 = st.tabs(["Application", "Details"])
    
    with tab1:
        code_editor_interface()
        output_container = st.container()
        with output_container:
            if st.session_state.edited_code:
                st.markdown("Generated App (by **Bad Scientist**):")
                execute_generated_code()
    
    with tab2:
        display_app_details()

if __name__ == "__main__":
    main()
import streamlit as st
import re
import importlib.util

st.set_page_config(
    page_title="Generated Apps",
    page_icon="🧪"
)

def extract_python_code(text: str) -> str:
    """
    Extracts Python code from markdown code blocks or raw text.
    
    Args:
        text (str): Text containing Python code
        
    Returns:
        str: Extracted Python code with proper whitespace
    """
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
    """
    Sanitizes the code string for safe execution.
    
    Args:
        code_string (str): Raw code string
        
    Returns:
        str: Sanitized code ready for execution
    """
    # Remove any page config calls
    code_string = re.sub(
        r'st\.set_page_config\([^)]*\)',
        '# Page config removed',
        code_string
    )
    
    # Ensure proper indentation
    lines = code_string.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.strip():  # Keep non-empty lines
            cleaned_lines.append(line.rstrip())  # Remove trailing whitespace
    
    return '\n'.join(cleaned_lines)

def format_code_for_display(code: str) -> str:
    """
    Formats code for display in markdown.
    
    Args:
        code (str): Code to format
        
    Returns:
        str: Formatted code in markdown
    """
    return f"```python\n{code}\n```"

def load_and_run_generated_code(code_string: str):
    """
    Safely loads and runs the generated code in the current Streamlit context.
    
    Args:
        code_string (str): Code to execute
    """
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
        module.st = st
        
        # Execute the code
        exec(code_string, module.__dict__)
        
        # Run main function if it exists
        if hasattr(module, 'main'):
            module.main()
            
    except Exception as e:
        st.error("Error running generated application")
        with st.expander("Error Details", expanded=True):
            st.exception(e)
            st.subheader("Code that failed:")
            st.code(code_string, language="python")

def main():
    st.title("Generated Application")
    
    if st.session_state.get('generated_code') is None:
        st.warning("No application has been generated yet. Please go to the Generator page to create an application.")
        return
    
    tab1, tab2 = st.tabs(["Run Application", "View Details & Code"])
    
    with tab1:
        st.write("Running Generated Application...")
        if st.session_state.generated_code:
            if st.button("⚡ Run Application"):
                load_and_run_generated_code(st.session_state.generated_code)
    
    with tab2:
        if st.session_state.get('app_results'):
            # Technical Requirements
            with st.expander("Technical Requirements Analysis", expanded=True):
                st.subheader("Analysis Results")
                if requirements := st.session_state.app_results.get("requirements"):
                    st.markdown(requirements)
                else:
                    st.info("No requirements analysis available")
            
            # Data Analysis
            with st.expander("Data Analysis", expanded=True):
                st.subheader("Data Requirements")
                if data_analysis := st.session_state.app_results.get("data_analysis"):
                    st.markdown(data_analysis)
                else:
                    st.info("No data analysis available")
            
            # Implementation Patterns
            with st.expander("Implementation Patterns", expanded=False):
                st.subheader("Reference Patterns")
                if patterns := st.session_state.app_results.get("reference_patterns"):
                    if isinstance(patterns, dict):
                        for pattern_type, pattern in patterns.items():
                            st.markdown(f"### {pattern_type.title()} Patterns")
                            st.markdown(pattern)
                    else:
                        st.markdown(patterns)
                else:
                    st.info("No implementation patterns available")
            
            # Streamlit Components
            with st.expander("Streamlit Components", expanded=False):
                st.subheader("Component Analysis")
                if components := st.session_state.app_results.get("streamlit_components"):
                    st.markdown(components)
                else:
                    st.info("No component analysis available")
            
            # Generated Code
            with st.expander("Generated Code", expanded=False):
                st.subheader("Final Implementation")
                if code := st.session_state.generated_code:
                    formatted_code = format_code_for_display(code)
                    st.markdown(formatted_code)
                    if st.button("Copy Code", key="copy_code"):
                        st.code(code, language="python")
                else:
                    st.warning("No code has been generated")
            
            # Debug Information
            if debug_msgs := st.session_state.get('debug_messages'):
                with st.expander("Debug Information", expanded=False):
                    st.subheader("Debug Log")
                    for msg in debug_msgs:
                        st.text(msg)

if __name__ == "__main__":
    main()
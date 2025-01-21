from typing import Optional, Dict, Any
from dataclasses import dataclass
from snowflake.snowpark.session import Session
from snowflake.core import Root
from config import CONNECTION_PARAMETER, MODEL_NAME, MODEL_TEMPERATURE, STREAMLIT_TEMPLATE
from tools.text_to_sql import SnowflakeTableTool, RAGSQLGenerator
from tools.text_to_sklearn import SklearnImplementationTool, RAGSklearnGenerator
from tools.text_to_matplotlib import MatplotlibVisualizationTool, RAGPythonGenerator
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress all Snowflake-related INFO messages
snowflake_loggers = [
    'snowflake.core._root',
    'snowflake.connector.connection',
    'snowflake.connector.cursor',
    'snowflake.core._internal.telemetry',
    'snowflake.snowpark.session',
    'snowflake.core.cortex.search_service._generated.api_client'
]

for log_name in snowflake_loggers:
    logging.getLogger(log_name).setLevel(logging.WARNING)

@dataclass
class AnalysisResult:
    """Container for analysis results"""
    data_analysis: str
    sklearn_code: str
    visualization_code: str
    streamlit_code: str

class DataScienceFlow:
    def __init__(self, session: Optional[Session] = None):
        """Initialize with shared resources for better performance"""
        self.session = session or Session.builder.configs(CONNECTION_PARAMETER).create()
        self.root = Root(self.session)
        
        # Initialize all generators with shared session
        self.initialize_generators()
        self.initialize_tools()

    def initialize_generators(self):
        """Initialize all RAG generators with shared session"""
        self.sql_generator = RAGSQLGenerator(session=self.session)
        self.sklearn_generator = RAGSklearnGenerator(session=self.session)
        self.viz_generator = RAGPythonGenerator(session=self.session)

    def initialize_tools(self):
        """Initialize all tools with shared generators"""
        self.snowflake_tool = SnowflakeTableTool(
            snowpark_session=self.session,
            rag_generator=self.sql_generator,
            result_as_answer=True
        )
        
        self.sklearn_tool = SklearnImplementationTool(
            snowpark_session=self.session,
            rag_generator=self.sklearn_generator,
            result_as_answer=True
        )
        
        self.viz_tool = MatplotlibVisualizationTool(
            snowpark_session=self.session,
            rag_generator=self.viz_generator,
            result_as_answer=True
        )

    def get_data_analysis_prompt(self, base_prompt: str) -> str:
        """Generate specific prompt for data analysis"""
        return f"""
        Task: Generate Snowflake SQL queries and data loading functions for a dashboard.
        
        Requirements:
        {base_prompt}
        
        Expected Output:
        1. SQL queries to extract relevant data
        2. Python functions to load and transform the data
        3. Error handling for data loading
        4. Proper data type conversions
        5. Column name standardization (lowercase)
        
        Note: Focus on efficient queries and proper error handling.
        """

    def get_ml_prompt(self, base_prompt: str, data_context: str) -> str:
        """Generate specific prompt for ML code"""
        return f"""
        Task: Generate a complete scikit-learn implementation for data analysis.
        
        Data Context:
        {data_context}
        
        Requirements:
        {base_prompt}
        
        Expected Output:
        1. Data preprocessing functions
        2. Feature engineering
        3. Model training pipeline
        4. Cross-validation
        5. Error handling
        6. Model evaluation metrics
        
        Note: Include docstrings and type hints.
        """

    def get_viz_prompt(self, base_prompt: str, data_context: str) -> str:
        """Generate specific prompt for visualization code"""
        return f"""
        Task: Generate matplotlib visualization functions for Streamlit dashboard.
        
        Data Context:
        {data_context}
        
        Requirements:
        {base_prompt}
        
        Expected Output:
        1. Functions to create each visualization
        2. Clear figure and axes management
        3. Proper color schemes
        4. Legend and label handling
        5. Error handling for missing data
        
        Note: Ensure all plots work with st.pyplot().
        """
    def analyze_data(self, prompt: str) -> str:
        """Analyze Snowflake data based on prompt"""
        logger.info("Analyzing Snowflake data...")
        return self.snowflake_tool.run(
            query=prompt,
            generate_sql=True
        )

    def generate_ml_code(self, prompt: str, data_context: str) -> str:
        """Generate sklearn implementation"""
        logger.info("Generating machine learning code...")
        return self.sklearn_tool.run(
            prompt=prompt,
            data_context=data_context
        )

    def generate_viz_code(self, prompt: str, data_context: str) -> str:
        """Generate visualization code"""
        logger.info("Generating visualization code...")
        return self.viz_tool.run(
            prompt=prompt,
            data_context=data_context
        )
    
    def generate_streamlit_code(
        self,
        data_analysis: str,
        viz_code: str
    ) -> str:
        """Generate Streamlit application code"""
        logger.info("Generating Streamlit application code...")

        prompt = f"""Revise this streamlit app to connect and retrieve data with snowflake and interactive visualization in one page streamlit app.

        This is the material current code and data explanation you need to combine:
        - Data needed or can be user:
        {data_analysis}

        - visualization idea:
        {viz_code}
        
        This is the current code and data explanation:

        {STREAMLIT_TEMPLATE}
        """   

        messages = json.dumps([
            {
                'role': 'system', 
                'content': """You are a seasoned Python and streamlit Developer. Generate a production-ready, error-free, streamlit app that can connect to Snowflake database and can create visualization using matplotlib (in streamlit we can display using st.pyplot). 
                If credentials needed, assume it already stored in the .env file. Return only a python code without any explanation"""
            },
            {
                'role': 'user', 
                'content': prompt
            }
        ])

        parameters = json.dumps({                               
            'temperature': MODEL_TEMPERATURE,
        })
        
        result = self.session.sql(
            "SELECT snowflake.cortex.complete(?, parse_json(?), parse_json(?))",
            params=[MODEL_NAME, messages, parameters]
        ).collect()[0][0]

        response = json.loads(result)
        
        if response and 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['messages'].strip()

        return response

    def execute_flow(self, prompt: str) -> AnalysisResult:
        """Execute flow with parallel processing where possible"""
        try:
            # Step 1: Get data analysis first (required for others)
            data_analysis = self.analyze_data(self.get_data_analysis_prompt(prompt))
            
            # Step 2: Run ML and viz generation in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_ml = executor.submit(
                    self.generate_ml_code,
                    self.get_ml_prompt(prompt, data_analysis),
                    data_analysis
                )
                
                future_viz = executor.submit(
                    self.generate_viz_code,
                    self.get_viz_prompt(prompt, data_analysis),
                    data_analysis
                )
                
                # Wait for both to complete
                sklearn_code = future_ml.result()
                viz_code = future_viz.result()
            
            # Step 3: Generate Streamlit app
            streamlit_code = self.generate_streamlit_code(
                data_analysis=data_analysis,
                viz_code=viz_code
            )
            
            return AnalysisResult(
                data_analysis=data_analysis,
                sklearn_code=sklearn_code,
                visualization_code=viz_code,
                streamlit_code=streamlit_code
            )

        except Exception as e:
            logger.error(f"Error in flow execution: {str(e)}")
            raise
        finally:
            # Clean up
            self.session.close()

# # USAGE

# def main():
#     # Example with improved prompts
#     dashboard_prompt = """
#     Create an interactive driver analytics dashboard with:
    
#     Data Requirements:
#     1. All Driver performance metrics (ratings, completed rides)
#     2. Temporal analysis (daily/weekly/monthly trends)
#     3. Geographic distribution of drivers
#     4. Revenue and payment analytics
    
#     Visualization Requirements:
#     1. Time series plots for driver activity
#     2. Distribution plots for ratings
#     3. Heatmap for geographic concentration
#     4. Interactive filters for time period and driver status
    
#     Technical Requirements:
#     1. Efficient data loading from Snowflake
#     2. Proper error handling
#     3. Interactive matplotlib plots
#     4. Clean and documented code
#     """
    
#     flow = DataScienceFlow()
#     result = flow.execute_flow(dashboard_prompt)

# if __name__ == "__main__":
#     main()
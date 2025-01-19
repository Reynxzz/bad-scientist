from typing import Type, Dict, Optional, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from snowflake.snowpark.session import Session
from config import MODEL_NAME
from dataclasses import dataclass
from snowflake.core import Root

class MatplotlibInput(BaseModel):
    """Input schema for Python visualization code generation."""
    prompt: str = Field(description="The prompt to generate matplotlib/seaborn visualization code")
    data_context: str = Field(
        default="",
        description="Context about available data/columns from previous SnowflakeTableTool"
    )

class RAGPythonGenerator:
    def __init__(self, session: Session, model_name: str = "mistral-large2", num_examples: int = 3):
        """Initialize the RAG Python generator
        
        Args:
            session: Snowflake session
            model_name: Name of the Cortex model to use
            num_examples: Number of examples to retrieve
        """
        self.session = session
        self.root = Root(session)
        self.model_name = model_name
        self.num_examples = num_examples

    def retrieve_examples(self, query: str) -> List[Dict]:
        """Retrieve similar examples from Cortex Search
        
        Args:
            query: Natural language query
        """
        search_service = (
            self.root
            .databases[self.session.get_current_database()]
            .schemas[self.session.get_current_schema()]
            .cortex_search_services['plt_code_search_svc']
        )

        response = search_service.search(
            query=query,
            columns=["prompt_text", "python_code"],
            limit=self.num_examples
        )

        return response.results if response.results else []

    def create_prompt(self, question: str, examples: List[Dict], data_context: str = "") -> str:
        """Create prompt for Python code generation"""
        prompt_text = f"""[INST]
        As an expert Python programmer, generate matplotlib/seaborn visualization code for the following task:

        Task: {question}

        Available Data Context:
        {data_context}

        Here are some similar examples to help guide you:
        """

        for i, example in enumerate(examples, 1):
            prompt_text += f"""Example {i}:
            Task: {example['prompt_text']}
            Python Code:
            ```python
            {example['python_code']}```
            """

        prompt_text += """
        Based on these examples and the available data context, generate Python visualization code for the original task:
        {question}

        Output only the Python code without any explanation or additional text.
        [/INST]"""

        return prompt_text.format(question=question, data_context=data_context)

    def run_cortex_complete(self, prompt: str) -> str:
        """Run Cortex Complete model
        
        Args:
            prompt: Input prompt for the model
        """
        result = self.session.sql(
            "SELECT snowflake.cortex.complete(?, ?)",
            params=[self.model_name, prompt]
        ).collect()[0][0]
        return result

    def generate_python(self, question: str, data_context: str = "") -> Dict:
        """Generate Python code for a given question
        
        Args:
            question: Natural language question
            data_context: Data context information
        """
        # Retrieve similar examples
        examples = self.retrieve_examples(question)
        
        # Create prompt with examples
        prompt = self.create_prompt(question, examples)
        
        # Generate Python code using Cortex Complete
        generated_code = self.run_cortex_complete(prompt).strip()
        
        # Clean up the generated code if it contains markdown markers
        if generated_code.startswith("```python"):
            generated_code = generated_code.replace("```python", "").replace("```", "").strip()
        
        return {
            'question': question,
            'data_context': data_context,
            'generated_code': generated_code,
            'examples_used': examples,
            'prompt_used': prompt
        }
    
class MatplotlibVisualizationTool(BaseTool):
    name: str = "Generate Matplotlib Visualization Code"
    description: str = """Generate Python code using matplotlib/seaborn for data visualization 
    based on natural language prompts and available data context. Uses RAG to find similar 
    examples and generate appropriate visualization code."""
    args_schema: Type[BaseModel] = MatplotlibInput
    
    def __init__(
        self, 
        snowpark_session: Session, 
        rag_generator: Optional['RAGPythonGenerator'] = None,
        result_as_answer: bool = False
    ):
        """Initialize the visualization tool
        
        Args:
            snowpark_session: Snowflake session
            rag_generator: Optional RAGPythonGenerator instance
            result_as_answer: Whether to return result as an answer
        """
        super().__init__()
        self._session = snowpark_session
        self._rag_generator = rag_generator or RAGPythonGenerator(session=snowpark_session)
        self.result_as_answer = result_as_answer

    def format_output(self, code: str, data_context: str) -> str:
        """Format the output code with proper markdown and context"""
        return code
    
    def validate_input(self, prompt: str, data_context: str) -> None:
        """Validate input parameters"""
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        if len(prompt.split()) < 3:
            raise ValueError("Prompt is too short. Please provide a more detailed description")

    def _run(self, prompt: str, data_context: str = "") -> str:
        """Generate matplotlib/seaborn visualization code"""
        try:
            print(f"MatplotlibVisualizationTool executing with prompt: {prompt}")
            print(f"Data context: {data_context}")

            # Validate inputs
            self.validate_input(prompt, data_context)

            # Generate visualization code
            result = self._rag_generator.generate_python(
                question=prompt,
                data_context=data_context  # Pass data_context
            )

            # Format and return the result
            return self.format_output(
                code=result['generated_code'],
                data_context=data_context
            )

        except Exception as e:
            error_message = f"Error generating visualization code: {str(e)}"
            print(error_message)
            raise RuntimeError(error_message)

    def run(self, prompt: str, data_context: str = "") -> str:
        """Public method to run the tool with error handling"""
        try:
            return self._run(prompt, data_context)
        except Exception as e:
            if self.result_as_answer:
                return f"Failed to generate visualization code: {str(e)}"
            raise


# # USAGE
# from snowflake.snowpark.session import Session
# from config import CONNECTION_PARAMETER

# session = Session.builder.configs(CONNECTION_PARAMETER).create()


# import time
# from datetime import datetime
# start_time = time.time()
# print(f"Starting execution at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# viz_tool = MatplotlibVisualizationTool(
#     snowpark_session=session,
#     result_as_answer=True
# )

# data_context = """
# Table: sales_data
# Columns:
# - date (DATE): Sale date
# - product_id (INT): Product identifier
# - revenue (FLOAT): Total revenue
# - units_sold (INT): Number of units sold
# """

# result = viz_tool.run(
#     prompt="Create a line plot showing daily revenue over time with a 7-day moving average",
#     data_context=data_context
# )

# end_time = time.time()
# execution_time = end_time - start_time

# print("\n=== Execution Results ===")
# print(f"Total execution time: {execution_time:.2f} seconds")
# print("\n=== Matplotlib Results ===")
# print(result)

# # # TESTED --> Total execution time: 9.08 seconds


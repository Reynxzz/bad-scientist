from typing import Type, Dict, Optional, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from snowflake.snowpark.session import Session
from config import MODEL_NAME, MODEL_TEMPERATURE
from snowflake.core import Root
import json

class SklearnInput(BaseModel):
    """Input schema for Python sklearn implementation code generation."""
    prompt: str = Field(description="The prompt to generate sklearn implementation code")
    data_context: str = Field(
        default="",
        description="Context about available data/columns"
    )

class RAGSklearnGenerator:
    def __init__(self, session: Session, model_name: str = MODEL_NAME, num_examples: int = 3):
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
            .cortex_search_services['sklearn_code_search_svc']
        )

        response = search_service.search(
            query=query,
            columns=["input", "output", "instruction"],
            limit=self.num_examples
        )
        return response.results if response.results else []

    def create_prompt(self, question: str, examples: List[Dict], data_context: str = "") -> str:
        """Create prompt for Python code generation"""
        prompt_text = f"""[INST]
        As an expert Python programmer, generate sklearn implementation code for the following task:

        Task: {question}

        Available Data Context:
        {data_context}

        Here are some similar examples to help guide you:
        """

        for i, example in enumerate(examples, 1):
            prompt_text += f"""Example {i}:
            Task: {example['input'][:200] + '...' if len(example['input']) > 200 else example['input']}
            Python Code:
            ```python
            {example['output']}```
            """

        prompt_text += f"""
        Based on these examples and the available data context, generate Python sklearn code for the original task:
        {question}

        Output only the Python code without any explanation or additional text.
        [/INST]"""


        return prompt_text

    def run_cortex_complete(self, prompt: str) -> str:
        """Run Cortex Complete model
        
        Args:
            prompt: Input prompt for the model
        """
        messages = json.dumps([
            {
                'role': 'system', 
                'content': 'You are a helpful AI assistant to implement scikit-learn in python using user specified data, dont use provided implementation example as it is but adapt it into user data to generate new sklearn implementation'
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
            params=[self.model_name, messages, parameters]
        ).collect()[0][0]

        response = json.loads(result)
        
        # Extract just the messages content from the first choice
        if response and 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['messages'].strip()

    def generate_python(self, question: str, data_context: str = "") -> Dict:
        """Generate Python code for a given question
        
        Args:
            question: Natural language question
            data_context: Data context information
        """
        examples = self.retrieve_examples(question)
        prompt = self.create_prompt(question, examples)
        generated_code = self.run_cortex_complete(prompt).strip()
        
        if generated_code.startswith("```python"):
            generated_code = generated_code.replace("```python", "").replace("```", "").strip()
        
        return {
            'question': question,
            'data_context': data_context,
            'generated_code': generated_code,
            'examples_used': examples,
            'prompt_used': prompt
        }
    
class SklearnImplementationTool(BaseTool):
    name: str = "Generate Sklearn implementation Code"
    description: str = """Generate Python code using scikit-learn (sklearn) for data regression or classification 
    based on natural language prompts and available data context. Uses RAG to find similar 
    examples and generate appropriate sklearn code."""
    args_schema: Type[BaseModel] = SklearnInput
    
    def __init__(
        self, 
        snowpark_session: Session, 
        rag_generator: Optional['RAGSklearnGenerator'] = None,
        result_as_answer: bool = False
    ):
        """Initialize the generation tool
        
        Args:
            snowpark_session: Snowflake session
            rag_generator: Optional RAGSklearnGenerator instance
            result_as_answer: Whether to return result as an answer
        """
        super().__init__()
        self._session = snowpark_session
        self._rag_generator = rag_generator or RAGSklearnGenerator(session=snowpark_session)
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
        """Generate sklearn implementation code"""
        try:
            print(f"SklearnImplementationTool executing with prompt: {prompt}")
            print(f"Data context: {data_context}")

            self.validate_input(prompt, data_context)

            result = self._rag_generator.generate_python(
                question=prompt,
                data_context=data_context
            )

            return self.format_output(
                code=result['generated_code'],
                data_context=data_context
            )

        except Exception as e:
            error_message = f"Error generating code: {str(e)}"
            print(error_message)
            raise RuntimeError(error_message)

    def run(self, prompt: str, data_context: str = "") -> str:
        """Public method to run the tool with error handling"""
        try:
            return self._run(prompt, data_context)
        except Exception as e:
            if self.result_as_answer:
                return f"Failed to generate code: {str(e)}"
            raise

# # USAGE
# from snowflake.snowpark.session import Session
# from config import CONNECTION_PARAMETER

# session = Session.builder.configs(CONNECTION_PARAMETER).create()

# import time
# from datetime import datetime
# start_time = time.time()
# print(f"Starting execution at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# viz_tool = SklearnImplementationTool(
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
#     prompt="Create a scikit learn implementation for regression task using this table",
#     data_context=data_context
# )

# end_time = time.time()
# execution_time = end_time - start_time

# print("\n=== Execution Results ===")
# print(f"Total execution time: {execution_time:.2f} seconds")
# print("\n=== Sklearn Results ===")
# print(result)

# # # TESTED --> Total execution time: 27.15 seconds


from typing import Type, Dict, Optional, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from snowflake.snowpark.session import Session
from config import MODEL_NAME
from dataclasses import dataclass
from snowflake.core import Root

@dataclass
class TableInfo:
    """Container for table information"""
    name: str
    description: str
    columns: Dict[str, Dict[str, str]]  # column_name -> {type, description, nullable}

class SnowflakeTableInput(BaseModel):
    """Input schema for table search and query generation."""
    query: str = Field(description="The search query to find relevant tables and generate SQL")
    generate_sql: bool = Field(
        default=False, 
        description="Whether to generate SQL query for the given input"
    )


class RAGSQLGenerator:
    def __init__(self, session: Session, model_name: str = "mistral-large2", num_examples: int = 3):
        """Initialize the RAG SQL generator
        
        Args:
            session: Snowflake session
            model_name: Name of the Cortex model to use
            num_examples: Number of examples to retrieve
        """
        self.session = session
        self.root = Root(session)
        self.model_name = model_name
        self.num_examples = num_examples

    def retrieve_examples(self, query: str, table_context: str = None) -> List[Dict]:
        """Retrieve similar examples from Cortex Search
        
        Args:
            query: Natural language query
            table_context: SQL table/schema information
        """
        search_service = (
            self.root
            .databases[self.session.get_current_database()]
            .schemas[self.session.get_current_schema()]
            .cortex_search_services['sql_query_search_svc']
        )

        search_query = query
        if table_context:
            search_query = f"{query} {table_context}"

        response = search_service.search(
            query=search_query,
            columns=["prompt_text", "sql_query", "sql_context", "sql_explanation"],
            limit=self.num_examples
        )

        return response.results if response.results else []

    def create_prompt(self, question: str, table_context: str, examples: List[Dict]) -> str:
        """Create prompt for SQL generation
        
        Args:
            question: Natural language question
            table_context: Table/schema information
            examples: Similar examples for few-shot learning
        """
        prompt = f"""[INST]
        As an expert SQL engineer, generate a Snowflake SQL query for the following question:

        Question: {question}

        Table Information:
        {table_context}

        Here are some similar examples to help guide you:

        """
        for i, example in enumerate(examples, 1):
            prompt += f"""Example {i}:
            Question: {example['prompt_text']}
            Table Information: {example.get('sql_context', '')}
            SQL: {example['sql_query']}
            Explanation: {example.get('sql_explanation', '')}

            """

        prompt += f"""
        Based on these examples and the provided table information, generate a Snowflake SQL query for the original question:
        {question}

        Output only the SQL query without any explanation or additional text.
        [/INST]"""

        return prompt

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

    def generate_sql(self, question: str, table_context: str = None) -> Dict:
        """Generate SQL for a given question with table context
        
        Args:
            question: Natural language question
            table_context: Table/schema information
        """
        examples = self.retrieve_examples(question, table_context)
        
        prompt = self.create_prompt(question, table_context or "", examples)
        
        generated_sql = self.run_cortex_complete(prompt)
        
        return {
            'question': question,
            'table_context': table_context,
            'generated_sql': generated_sql,
            'examples_used': examples,
            'prompt_used': prompt
        }

class SnowflakeTableTool(BaseTool):
    name: str = "Search and Query Snowflake Tables"
    description: str = """Search available Snowflake tables, their structures, and optionally 
    generate SQL queries to analyze the data. Provides information about relevant tables 
    and their columns, along with SQL queries for implementation."""
    args_schema: Type[BaseModel] = SnowflakeTableInput
    
    def __init__(
        self, 
        snowpark_session: Session, 
        rag_generator: Optional['RAGSQLGenerator'] = None,
        result_as_answer: bool = False
    ):
        super().__init__()
        self._session = snowpark_session
        self._rag_generator = rag_generator
        self.result_as_answer = result_as_answer

    def _get_tables_info(self) -> Dict[str, TableInfo]:
        """Retrieve and organize table and column information."""
        tables_info = {}
        
        tables_raw = self._session.sql("""
            SELECT 
                table_name,
                table_type,
                comment
            FROM information_schema.tables 
            WHERE table_schema = CURRENT_SCHEMA()
        """).collect()

        columns_raw = self._session.sql("""
            SELECT 
                table_name,
                column_name,
                data_type,
                is_nullable,
                comment
            FROM information_schema.columns
            WHERE table_schema = CURRENT_SCHEMA()
            ORDER BY table_name, ordinal_position
        """).collect()

        for table in tables_raw:
            table_name = table['TABLE_NAME']
            tables_info[table_name] = TableInfo(
                name=table_name,
                description=table['COMMENT'] if table['COMMENT'] else 'No description available',
                columns={}
            )

            table_columns = [col for col in columns_raw if col['TABLE_NAME'] == table_name]
            for col in table_columns:
                tables_info[table_name].columns[col['COLUMN_NAME']] = {
                    'type': col['DATA_TYPE'],
                    'nullable': col['IS_NULLABLE'] == 'YES',
                    'description': col['COMMENT'] if col['COMMENT'] else 'No description'
                }

        return tables_info

    def _format_table_context(self, tables_info: Dict[str, TableInfo]) -> str:
        """Format tables information into a structured context string."""
        context_parts = []
        
        for table_info in tables_info.values():
            columns_text = "\n".join([
                f"- {col_name} ({details['type']}){'[nullable]' if details['nullable'] else ''}: {details['description']}"
                for col_name, details in table_info.columns.items()
            ])
            
            context_parts.append(f"""
            Table: {table_info.name}
            Description: {table_info.description}
            Columns:
            {columns_text}
            """)

        return "\n\n".join(context_parts)

    def _get_relevant_tables(self, query: str, tables_context: str) -> str:
        """Use LLM to identify relevant tables for the query."""
        prompt = f"""
        Based on the following Snowflake tables information, recommend which tables and columns 
        would be most relevant for the given query. Be concise and clear.

        Available Tables Information:
        {tables_context}

        Query: {query}

        Please provide:
        1. Most relevant tables for this query
        2. Key columns that could be useful, with their data types
        3. Brief explanation of why these tables/columns are relevant

        If no related/relevant tables are found, respond with: "No Snowflake data required".
        """
        
        return self._session.sql(
            "SELECT snowflake.cortex.complete(?, ?)",
            params=(MODEL_NAME, prompt)
        ).collect()[0][0]

    def _run(self, query: str, generate_sql: bool = False) -> str:
        """Search for relevant tables and optionally generate SQL query."""
        print(f"`SnowflakeTableTool` called with query: {query}, generate_sql: {generate_sql}")

        tables_info = self._get_tables_info()
        tables_context = self._format_table_context(tables_info)
        
        tables_analysis = self._get_relevant_tables(query, tables_context)
        
        if "No Snowflake data required" in tables_analysis:
            return tables_analysis
            
        generated_sql = ""
        if generate_sql and self._rag_generator:
            try:
                sql_result = self._rag_generator.generate_sql(
                    question=query,
                    table_context=tables_context
                )
                generated_sql = sql_result['generated_sql']
            except Exception as e:
                generated_sql = f"\n\nSQL Generation Error: {str(e)}"
        
        return f"{tables_analysis}{generated_sql}"

# # Usage:
# from snowflake.snowpark.session import Session
# from config import CONNECTION_PARAMETER

# session = Session.builder.configs(CONNECTION_PARAMETER).create()

# rag_generator = RAGSQLGenerator(session=session)

# snowflake_tool = SnowflakeTableTool(
#     snowpark_session=session,
#     rag_generator=rag_generator
# )

# import time
# from datetime import datetime

# start_time = time.time()
# print(f"Starting execution at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# result = snowflake_tool.run(
#     query="Create a Streamlit dashboard for Uber's operations team to analyze driver performance and payments",
#     generate_sql=True
# )

# end_time = time.time()
# execution_time = end_time - start_time

# print("\n=== Execution Results ===")
# print(f"Total execution time: {execution_time:.2f} seconds")
# print("\n=== Query Results ===")
# print(result)

# # TESTED --> Total execution time: 28.17 seconds
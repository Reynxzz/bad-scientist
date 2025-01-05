from typing import Type, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from snowflake.snowpark.session import Session

class SnowflakeTableInput(BaseModel):
    """Input schema for table search."""
    query: str = Field(description="The search query to find relevant tables")

class SnowflakeTableTool(BaseTool):
    name: str = "Search Snowflake Tables"
    description: str = """Search available Snowflake tables and their structures.
    Provides information about relevant tables and their columns for implementation."""
    args_schema: Type[BaseModel] = SnowflakeTableInput
    
    def __init__(self, snowpark_session: Session, result_as_answer: bool = False):
        super().__init__()
        self._session = snowpark_session
        self.result_as_answer = result_as_answer

    def _run(self, query: str) -> str:
        """Search for relevant tables and summarize with LLM."""
        # Get tables info
        tables_info = self._session.sql("""
            SELECT 
                table_name,
                table_type,
                comment
            FROM information_schema.tables 
            WHERE table_schema = CURRENT_SCHEMA()
        """).collect()

        # Get columns info for all tables
        columns_info = self._session.sql("""
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

        # Format table information
        context_parts = []
        
        for table in tables_info:
            table_name = table['TABLE_NAME']
            table_comment = table['COMMENT'] if table['COMMENT'] else 'No description available'
            
            # Get columns for this table
            table_columns = [
                col for col in columns_info 
                if col['TABLE_NAME'] == table_name
            ]
            
            columns_text = "\n".join([
                f"- {col['COLUMN_NAME']} ({col['DATA_TYPE']}){'[nullable]' if col['IS_NULLABLE'] == 'YES' else ''}: {col['COMMENT'] if col['COMMENT'] else 'No description'}"
                for col in table_columns
            ])
            
            context_parts.append(f"""
Table: {table_name}
Description: {table_comment}
Columns:
{columns_text}
""")

        context = "\n\n".join(context_parts)
        
        # Generate LLM summary
        prompt = f"""
        Based on the following Snowflake tables information, recommend which tables and columns would be most relevant for the given query.
        Make it short and clear in less than 50 words for each table suggested.

        Available Tables Information:
        {context}

        Query: {query}

        Please provide:
        1. Most relevant tables for this query
        2. Key columns that could be useful
        3. Brief explanation of how these could be used
        """
        
        response = self._session.sql(
            "SELECT snowflake.cortex.complete(?, ?)",
            params=("mistral-large2", prompt)
        ).collect()[0][0]

        print("Snowflake Tool Response:", response)

        return response
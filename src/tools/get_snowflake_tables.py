from snowflake.snowpark.session import Session
from pydantic import BaseModel, Field, PrivateAttr
from crewai_tools.tools.base_tool import BaseTool
from typing import List, Type
import pandas as pd

class TableInfoInput(BaseModel):
    """Input schema for table information search."""
    query: str = Field(description="The search query to find relevant tables")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{"query": "customer transaction data with timestamps"}]
        }
    }

class TableInfo(BaseModel):
    """Schema for table information."""
    table_name: str = Field(description="Name of the table")
    columns: List[str] = Field(description="List of column names")
    sample_data: str = Field(description="Sample data from the table in string format")

class TableInfoOutput(BaseModel):
    """Output schema for table information search."""
    tables: List[TableInfo] = Field(description="List of relevant tables and their information")

class SnowflakeTableTool(BaseTool):
    """Tool for searching and analyzing Snowflake tables."""
    name: str = "Search Snowflake Tables"
    description: str = "Search through available Snowflake tables and analyze their structure and content"
    args_schema: Type[BaseModel] = TableInfoInput
    
    _session: Session = PrivateAttr()
    
    def __init__(self, snowpark_session: Session):
        super().__init__()
        self._session = snowpark_session
    
    def _run(self, query: str, **kwargs) -> dict:
        """Search for relevant tables based on the query."""
        try:
            # Get list of all tables in current database and schema
            tables_df = self._session.sql("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = CURRENT_SCHEMA()
            """).collect()
            
            relevant_tables = []
            for table_row in tables_df:
                table_name = table_row['TABLE_NAME']
                
                # Get column information
                columns_df = self._session.sql(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                """).collect()
                
                columns = [row['COLUMN_NAME'] for row in columns_df]
                
                # Get sample data
                sample_data = self._session.sql(f"""
                    SELECT * FROM {table_name} LIMIT 1
                """).collect()
                
                sample_str = pd.DataFrame(sample_data).to_string()
                
                relevant_tables.append({
                    "table_name": table_name,
                    "columns": columns,
                    "sample_data": sample_str
                })
            
            return {"tables": relevant_tables}
            
        except Exception as e:
            print(f"Error in SnowflakeTableTool: {str(e)}")
            return {"tables": []}

    def _parse_input(self, query: str) -> str:
        """Parse and validate the input."""
        if isinstance(query, dict) and 'query' in query:
            return query['query']
        return query
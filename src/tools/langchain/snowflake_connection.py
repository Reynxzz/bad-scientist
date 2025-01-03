"""
Snowflake connection management module.
"""

import os
from snowflake.snowpark.session import Session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SnowflakeConnection:
    """Manages Snowflake connection configuration and session creation."""
    
    def __init__(self):
        self.connection_params = {
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_USER_PASSWORD"),
            "role": os.getenv("SNOWFLAKE_ROLE"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE")
        }
        
    def create_session(self) -> Session:
        """Create and return a new Snowflake session."""
        return Session.builder.configs(self.connection_params).create()
"""
Document search service using Snowflake's search capabilities.
"""

from typing import Dict, List
from snowflake.core import Root
from snowflake.snowpark.session import Session

class SearchService:
    """Handles document search operations using Snowflake's search services."""
    
    def __init__(self, root: Root, session: Session):
        self.root = root
        self.session = session
        
    def search_documents(self, service_name: str, query: str, 
                        columns: List[str], limit: int = 5) -> Dict[str, List[Dict[str, str]]]:
        """
        Search documents using specified search service.
        
        Args:
            service_name: Name of the search service
            query: Search query string
            columns: List of columns to retrieve
            limit: Maximum number of results to return
            
        Returns:
            Dictionary containing search results
        """
        search_service = (
            self.root
            .databases[self.session.get_current_database()]
            .schemas[self.session.get_current_schema()]
            .cortex_search_services[service_name]
        )
        
        results = search_service.search(
            query=query,
            columns=columns,
            limit=limit
        )
        
        return {"results": [dict(r) for r in results.results]}
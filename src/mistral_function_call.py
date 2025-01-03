from snowflake.snowpark.session import Session
from snowflake.core import Root
import os
from mistralai import Mistral, UserMessage
from typing import Dict, Any
import json

# Initialize Snowflake session
connection_params = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_USER_PASSWORD"),
    "role": os.getenv("SNOWFLAKE_ROLE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE")
}

snowpark_session = Session.builder.configs(connection_params).create()
root = Root(snowpark_session)

# Define the search functions
def search_requirements(query: str) -> Dict[str, Any]:
    """Search through requirements documents."""
    service_name = "req_docs_search_svc"
    search_service = (
        root
        .databases[snowpark_session.get_current_database()]
        .schemas[snowpark_session.get_current_schema()]
        .cortex_search_services[service_name]
    )
    
    results = search_service.search(
        query=query,
        columns=["doc_text", "source"],
        limit=5
    )
    
    return {
        "results": [
            {"doc_text": r['doc_text'], "source": r['source']}
            for r in results.results
        ]
    }

def search_technical_docs(query: str, tech_stack: str) -> Dict[str, Any]:
    """Search through technical documentation."""
    service_name = f"{tech_stack}_docs_search_svc"
    search_service = (
        root
        .databases[snowpark_session.get_current_database()]
        .schemas[snowpark_session.get_current_schema()]
        .cortex_search_services[service_name]
    )
    
    results = search_service.search(
        query=query,
        columns=["doc_text"],
        limit=5
    )
    
    return {
        "results": [
            {"doc_text": r['doc_text'], "source": tech_stack}
            for r in results.results
        ]
    }

# Define function schemas for Mistral
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_requirements",
            "description": "Search through business requirements documents",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to use"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_technical_docs",
            "description": "Search through technical documentation for specific tech stack",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to use"
                    },
                    "tech_stack": {
                        "type": "string",
                        "enum": ["streamlit", "sklearn"],
                        "description": "Technology stack to search"
                    }
                },
                "required": ["query", "tech_stack"]
            }
        }
    }
]

# Map function names to their implementations
function_map = {
    "search_requirements": search_requirements,
    "search_technical_docs": search_technical_docs
}

def process_rag_query(query: str, verbose=True):
    """Process a RAG query using Mistral and Snowflake Cortex search.
    
    Args:
        query: The user's query
        verbose: If True, prints detailed information about tool usage and results
    """
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    
    # Initial query to Mistral
    messages = [
        {
            "role": "user",
            "content": query
        }
    ]
    # Alternatively with UserMessage:
    # messages = [UserMessage(content=query)]
    
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=messages,
        tools=tools,
        tool_choice="any"
    )
    
    # If Mistral wants to use a tool
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        
        # Get function details
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        # Execute the function
        if verbose:
            print("\n=== Tool Execution Details ===")
            print(f"Using function: {function_name}")
            print(f"Function arguments: {json.dumps(function_args, indent=2)}")
            
        function_result = function_map[function_name](**function_args)
        
        if verbose:
            print("\n=== Search Results ===")
            print(json.dumps(function_result, indent=2))
            print("========================")
        # Add tool response to messages
        messages.extend([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "content": json.dumps(function_result),
                "tool_call_id": tool_call.id,
                "name": function_name
            }
        ])
        
        # Get final response from Mistral
        final_response = client.chat.complete(
            model="mistral-large-latest",
            messages=messages
        )
        
        return final_response.choices[0].message.content
    
    return response.choices[0].message.content

# Example usage
if __name__ == "__main__":
    # Example query
    # example_query = "Find information about using TF-IDF in scikit-learn and how to code in python"
    example_query = "What are the development timeline requirements for the project?"
    result = process_rag_query(example_query, verbose=True)
    print(f"\nQuery: {example_query}")
    print(f"Response: {result}")
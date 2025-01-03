# # from litellm import completion

# # response = completion(
# #     model="mistral/mistral-large-2407", 
# #     # model = "gemini/gemini-1.5-flash",
# #     messages=[
# #        {"role": "user", "content": "hello from litellm"}
# #    ],
# # )
# # print(response)
# # print(response.choices[0].message.tool_calls[0])
# # # print(response["choices"][0]["message"]["content"])

# from snowflake.snowpark.session import Session
# from snowflake.core import Root
# import os
# import litellm
# from typing import Dict, Any
# import json

# # [Snowflake initialization code remains the same]
# connection_params = {
#     "account": os.getenv("SNOWFLAKE_ACCOUNT"),
#     "user": os.getenv("SNOWFLAKE_USER"),
#     "password": os.getenv("SNOWFLAKE_USER_PASSWORD"),
#     "role": os.getenv("SNOWFLAKE_ROLE"),
#     "database": os.getenv("SNOWFLAKE_DATABASE"),
#     "schema": os.getenv("SNOWFLAKE_SCHEMA"),
#     "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE")
# }

# snowpark_session = Session.builder.configs(connection_params).create()
# root = Root(snowpark_session)

# # [Search functions remain the same]
# def search_requirements(query: str) -> Dict[str, Any]:
#     """Search through requirements documents."""
#     # [Function implementation remains the same]
#     service_name = "req_docs_search_svc"
#     search_service = (
#         root
#         .databases[snowpark_session.get_current_database()]
#         .schemas[snowpark_session.get_current_schema()]
#         .cortex_search_services[service_name]
#     )
    
#     results = search_service.search(
#         query=query,
#         columns=["doc_text", "source"],
#         limit=5
#     )
    
#     return {
#         "results": [
#             {"doc_text": r['doc_text'], "source": r['source']}
#             for r in results.results
#         ]
#     }

# def search_technical_docs(query: str, tech_stack: str) -> Dict[str, Any]:
#     """Search through technical documentation."""
#     # [Function implementation remains the same]
#     service_name = f"{tech_stack}_docs_search_svc"
#     search_service = (
#         root
#         .databases[snowpark_session.get_current_database()]
#         .schemas[snowpark_session.get_current_schema()]
#         .cortex_search_services[service_name]
#     )
    
#     results = search_service.search(
#         query=query,
#         columns=["doc_text"],
#         limit=5
#     )
    
#     return {
#         "results": [
#             {"doc_text": r['doc_text'], "source": tech_stack}
#             for r in results.results
#         ]
#     }

# # Define tools for function calling
# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "search_requirements",
#             "description": "Search through business requirements documents",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "query": {
#                         "type": "string",
#                         "description": "The search query to use"
#                     }
#                 },
#                 "required": ["query"]
#             }
#         }
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "search_technical_docs",
#             "description": "Search through technical documentation for specific tech stack",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "query": {
#                         "type": "string",
#                         "description": "The search query to use"
#                     },
#                     "tech_stack": {
#                         "type": "string",
#                         "enum": ["streamlit", "sklearn"],
#                         "description": "Technology stack to search"
#                     }
#                 },
#                 "required": ["query", "tech_stack"]
#             }
#         }
#     }
# ]

# # Map function names to their implementations
# function_map = {
#     "search_requirements": search_requirements,
#     "search_technical_docs": search_technical_docs
# }

# def process_rag_query(query: str, verbose=True):
#     """Process a RAG query using litellm and Snowflake Cortex search."""
    
#     # Initial query parameters
#     params = {
#         "model": "mistral/mistral-large-latest",
#         "messages": [{"role": "user", "content": query}],
#         "tools": tools,
#         "tool_choice": "auto",
#         "api_key": os.getenv("MISTRAL_API_KEY")
#     }
    
#     # Get initial response
#     response = litellm.completion(**params)
    
#     # If the model wants to use a tool
#     if response.choices[0].message.tool_calls:
#         tool_call = response.choices[0].message.tool_calls[0]
        
#         # Get function details
#         function_name = tool_call.function.name
#         function_args = json.loads(tool_call.function.arguments)
        
#         # Execute the function
#         if verbose:
#             print("\n=== Tool Execution Details ===")
#             print(f"Using function: {function_name}")
#             print(f"Function arguments: {json.dumps(function_args, indent=2)}")
            
#         function_result = function_map[function_name](**function_args)
        
#         if verbose:
#             print("\n=== Search Results ===")
#             print(json.dumps(function_result, indent=2))
#             print("========================")
        
#         # Add tool response to messages
#         params["messages"].extend([
#             {
#                 "role": "assistant",
#                 "content": None,
#                 "tool_calls": [
#                     {
#                         "id": tool_call.id,
#                         "type": "function",
#                         "function": {
#                             "name": function_name,
#                             "arguments": tool_call.function.arguments
#                         }
#                     }
#                 ]
#             },
#             {
#                 "role": "tool",
#                 "content": json.dumps(function_result),
#                 "tool_call_id": tool_call.id,
#                 "name": function_name
#             }
#         ])
        
#         # Get final response
#         final_response = litellm.completion(**params)
#         return final_response.choices[0].message.content
    
#     return response.choices[0].message.content

# # Example usage
# if __name__ == "__main__":
#     example_query = "What are the development timeline requirements for the project?"
#     result = process_rag_query(example_query, verbose=True)
#     print(f"\nQuery: {example_query}")
#     print(f"Response: {result}")


import litellm
import os

def get_weather(location: str, unit: str = "celsius"):
    # Mock weather function
    return f"The weather in {location} is 22 {unit}"

# Define the tool/function
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    }
}]

# Set up messages and parameters
messages = [{"role": "user", "content": "What's the weather like in Paris?"}]

# Initial call to Mistral
response = litellm.completion(
    model="mistral/mistral-large-2407",
    messages=messages,
    tools=tools,
    api_key=os.getenv("MISTRAL_API_KEY")
)

# Check if there's a function call
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    
    # Execute function
    function_args = eval(tool_call.function.arguments)
    function_response = get_weather(**function_args)
    
    # Add the function call and result to messages
    messages.extend([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call]
        },
        {
            "role": "tool",
            "content": function_response,
            "tool_call_id": tool_call.id,
            "name": "get_weather"
        }
    ])
    
    # Get final response
    final_response = litellm.completion(
        model="mistral/mistral-large-latest",
        messages=messages,
        api_key=os.getenv("MISTRAL_API_KEY")
    )
    
    print(final_response.choices[0].message.content)
else:
    print(response.choices[0].message.content)
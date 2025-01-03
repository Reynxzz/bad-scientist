from langchain.tools import Tool
from langchain_mistralai import ChatMistralAI
import os
from langchain.agents import  AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Initialize the Mistral model
chat_model = ChatMistralAI(
    mistral_api_key=os.environ["MISTRAL_API_KEY"],
    model="mistral-large-latest",
    temperature=0.7
)

def get_weather(location: str) -> str:
    """Get weather information for a location"""
    return f"The weather in {location} is 22°C"

# Create tools list
tools = [
    Tool(
        name="GetWeather",
        func=get_weather,
        description="Get current weather for a city"
    )
]

# Create the prompt template
prompt = PromptTemplate.from_template(
    """Answer the following question as best you can:

{input}

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""
)

# Create the agent
agent = create_react_agent(
    llm=chat_model,
    tools=tools,
    prompt=prompt
)

# Create the executor with error handling
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

try:
    response = agent_executor.invoke(
        {"input": "What's the weather like in Paris?"}
    )
    print("\nResponse:", response['output'])
except Exception as e:
    print(f"Error occurred: {str(e)}")
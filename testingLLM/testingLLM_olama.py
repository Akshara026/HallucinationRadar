# from langchain.tools import tool
# from langchain.agents import AgentExecutor, create_react_agent
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama import ChatOllama


# @tool
# def multiply(a: int, b: int) -> int:
#     """Multiply a and b."""
#     return a * b


# @tool
# def add(a: int, b: int) -> int:
#     """Add a and b."""
#     return a + b


# @tool
# def divide(a: int, b: int) -> float:
#     """Divide a and b."""
#     return a / b


# tools = [add, multiply, divide]


# llm = ChatOllama(
#     model="llama3",
#     temperature=0
# )


# prompt = ChatPromptTemplate.from_template("""
# You are a strict math assistant. Always use tools. Never guess.

# You have access to the following tools:
# {tools}

# IMPORTANT:
# - When calling a tool, you MUST use JSON format
# - Example:
#   Action: add
#   Action Input: {{"a": 10, "b": 5}}

# Use the following format:

# Question: {input}
# Thought: think about what to do
# Action: one of [{tool_names}]
# Action Input: JSON with correct arguments
# Observation: result of the tool
# ... (repeat as needed)
# Final Answer: the answer

# {agent_scratchpad}
# """)


# agent = create_react_agent(llm, tools, prompt)

# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     verbose=True
# )


# response = agent_executor.invoke({
#     "input": "What is (10 + 5) * 2?"
# })

# print("\nFinal Answer:", response["output"])

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel


class MathInput(BaseModel):
    a: int
    b: int


def add_func(a: int, b: int) -> int:
    return a + b


def multiply_func(a: int, b: int) -> int:
    return a * b


def divide_func(a: int, b: int) -> float:
    return a / b


add = StructuredTool.from_function(
    func=add_func, args_schema=MathInput, description="Add two numbers"
)

multiply = StructuredTool.from_function(
    func=multiply_func, args_schema=MathInput, description="Multiply two numbers"
)

divide = StructuredTool.from_function(
    func=divide_func, args_schema=MathInput, description="Divide two numbers"
)

tools = [add, multiply, divide]


llm = ChatOllama(model="llama3.1", temperature=0)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a strict math assistant. Always use tools."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


response = agent_executor.invoke({"input": "What is (10 + 5) * 2?"})

print("\nFinal Answer:", response["output"])

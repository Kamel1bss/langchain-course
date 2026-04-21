from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient

load_dotenv()

class Source(BaseModel):
    """Schema for a source used by the agent."""
    name: str
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent's response with answer and sources."""
    answer: str = Field(description="The answer provided by the agent to the user's query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

tavily = TavilyClient()  

@tool
def search(query: str) -> str:
    """
    Tool that searches the web.
    Args:
        query: The search query.
    Returns:
        The search results.
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query) 

llm = ChatOllama(temperature=0, model="llama3.1:8b")
tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="search for 3 postings for .net developer positions in Egypt on linkedin and list their details")]})
    print(f"Agent response: {result}")


if __name__ == "__main__":
    main()

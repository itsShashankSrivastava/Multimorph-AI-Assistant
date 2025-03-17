# if you dont use pipenv uncomment the following:
from dotenv import load_dotenv
load_dotenv()

#Step1: Setup API Keys for Groq, Anthropic (Claude), and Tavily
import os

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")

#Step2: Setup LLM & Tools
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_anthropic import ChatAnthropic  

anthropic_llm = ChatAnthropic(model="claude-3-5-sonnet-20241022") 

groq_llm = ChatGroq(model="llama-3.3-70b-versatile")

search_tool = TavilySearchResults(max_results=2)

#Step3: Setup AI Agent with Search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt = "Act as an AI chatbot who is smart and friendly"

# tools = [search_tool]
# agent = create_react_agent(
#     model=groq_llm,  
#     tools=tools,
#     state_modifier=system_prompt
# )

# query = "Tell me about the trends in crypto market"

# state = {"messages": query}
# response = agent.invoke(state)
# # print(response)
# messages = response.get("messages")
# ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
# print(ai_messages[-1])

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider == "Groq":
        llm = ChatGroq(model=llm_id)
    elif provider == "Anthropic":  
        llm = ChatAnthropic(model=llm_id)  

    tools = [search_tool] if allow_search else []
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )
    
    state = {"messages": query}
    response = agent.invoke(state)
    messages = response.get("messages")
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]

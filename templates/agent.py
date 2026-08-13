from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny in {city}"

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
checkpointer = InMemorySaver()
agent = create_agent(model, tools=[get_weather], checkpointer=checkpointer)

config = {"configurable": {"thread_id": "shivansh-session-1"}}

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break
    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]}, config)
    print("Agent:", result["messages"][-1].content[0]["text"])
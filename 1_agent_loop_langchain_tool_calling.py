from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

# Tools (functions) that the agent can call (lanchain tool decorator)
@tool
def get_product_price(product: str) -> float:
    """Get the price of a product in the catalog."""
    prices = {
        "laptop": 999.0,
        "smartphone": 499.0,
        "headphones": 199.0 
    }
    print(f"  >> Executing get_product_price(product='{product}')")
    return prices.get(product.lower(), 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the discounted price."""
    """Available discount tiers: Bronze, Silver, Gold"""
    discounts = {
        "bronze":5,
        "silver":12,
        "gold": 23
    }

    discount = discounts.get(discount_tier.lower(), 0)
    print(f"  >>> executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    return round(price * (1 - discount /100), 2)

############################ Agent loop

@traceable(name="langchain_agent_loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tool_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools) # only with Models with function tooling

    print(f"Question: {question}")
    print("=" * 80)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant"
                "You have access to a product catalog tool"
                "and a discount application tool.\n\n"
                "STRICT RULES: - you must follow these exactly :\n"
                "1. NEVER guess or assme any product price " 
                "you MUST call the get_product_price tool first to get the real price of a product.\n"
                "2. only call the apply_discount tool AFTER you have received "
                "a price from get_product_price. pass the exact price "
                "returned by get_product_price - do NOT pass a made-up numebr.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always you the apply_discount tool.\n"
                "4. IFf the user does not specify a discount tier, "
                "ask them which tier to use - do NOT assume a default tier."
            )
        ),
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print("Final answer:", ai_message.content)
            return ai_message.content
        
        tool_call = tool_calls[0]
           
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"[Tool selected]  {tool_name} with args: {tool_args}")

        tool_to_call = tool_dict.get(tool_name)
        if not tool_to_call:
            print(f">>> ERROR: Tool '{tool_name}' not found.")
            return None
        
        observation = tool_to_call.invoke(tool_args)

        print(f"[Tool result] {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print(">> ERROR: Max iterations reached without a final answer.")
    return None
         



if __name__ == "__main__":
    print("Hello langchain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
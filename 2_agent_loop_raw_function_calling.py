from dotenv import load_dotenv
from langchain.tools import tool
from langsmith import traceable

load_dotenv()
import ollama

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Get the price of a product in the catalog."""
    prices = {
        "laptop": 999.0,
        "smartphone": 499.0,
        "headphones": 199.0 
    }
    print(f"  >> Executing get_product_price(product='{product}')")
    return prices.get(product.lower(), 0)

@traceable(run_type="tool")
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


tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Get the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The name of the product., e.g. laptop, smartphone, headphones"
                    }
                },
                "required": ["product"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the discounted price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price."
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier to apply."
                    }
                },
                "required": ["price", "discount_tier"]
            }
        }
    }
]



@traceable(name="ollama chat", run_type="llm")
def ollama_chat_traced(messages):
    response = ollama.chat(MODEL, messages=messages, tools=tools_for_llm)
    return response

@traceable(name="ollama_agent_loop")
def run_agent(question: str):
    tool_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }

    print(f"Question: {question}")
    print("=" * 80)

    messages = [
        {
            "role": "system",
            "content": (
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
                "4. IF the user does not specify a discount tier, "
                "ask them which tier to use - do NOT assume a default tier."
            )
        },
        {"role": "user", "content": question}
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        response = ollama_chat_traced(messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print("Final answer:", ai_message.content)
            return ai_message.content

        tool_call = tool_calls[0]

        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"[Tool selected]  {tool_name} with args: {tool_args}")

        tool_to_use = tool_dict.get(tool_name)
        if not tool_to_use:
            print(f"Error: Tool '{tool_name}' not found.")
            return None

        observation = tool_to_use(**tool_args)

        print(f"[Tool result] {observation}")

        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation)
            }
        )


    print(">> ERROR: Max iterations reached without a final answer.")
    return None
         

if __name__ == "__main__":
    print("Hello langchain Agent (.bind_tools)!")
    run_agent("What is the price of a laptop after applying a gold discount?")
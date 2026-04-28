from dotenv import load_dotenv
from langsmith import traceable
import re
import inspect
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

    price = float(price)
    discount = discounts.get(discount_tier.lower(), 0)
    print(f"  >>> executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    return round(price * (1 - discount /100), 2)

############################ Agent loop
tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}


def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_func in tools_dict.items():
        original_func = getattr(tool_func, "__wrapped__", tool_func)
        signature = inspect.signature(original_func)
        doc_string = inspect.getdoc(original_func) or ""
        descriptions.append(f"{tool_name}{signature} - {doc_string}")
    return "\n".join(descriptions)


tool_descs = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_system_prompt = f"""
"STRICT RULES: - you must follow these exactly :
1. NEVER guess or assme any product price you MUST call the get_product_price tool first to get the real price of a product.
2. only call the apply_discount tool AFTER you have received a price from get_product_price. pass the exact price returned by get_product_price - do NOT pass a made-up numebr.
3. NEVER calculate discounts yourself using math. Always you the apply_discount tool.
4. IFf the user does not specify a discount tier, ask them which tier to use - do NOT assume a default tier

Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

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

Question: {{question}}
"""

@traceable(name="ollama chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    response = ollama.chat(model=model, messages=messages, options=options)
    return response

@traceable(name="ollama_agent_loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 80)

    prompt = react_system_prompt.format(question=question)
    scratchpad = ""


    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        full_prompt = prompt + "\n" + scratchpad 

        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}], 
            options={"stop": ["\nObservation"], "temperature": 0}
        )

        output = response.message.content

        print(f"LLM output:\n{output}")


        print(f" [Parsing]  Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f" [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 80)
            print(f"Final Answer: {final_answer}")
            return final_answer

        
        print(f" [Parsing] Looking for Action and Action Input...")


        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(" [Parsing] Error: Could not find Action/Action Input in LLM output.")
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"[Tool selected]  {tool_name} with args: {tool_input_raw}")

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"[Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list[str](tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))

        print(f"[Tool result] {observation}")

        scratchpad += f"{output}\nObservation: {observation}\nThought:"
       


    print(">> ERROR: Max iterations reached without a final answer.")
    return None
         

if __name__ == "__main__":
    print("Hello langchain Agent (.bind_tools)!")
    run_agent("What is the price of a laptop after applying a gold discount?")
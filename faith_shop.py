from typing import List

from langchain_core.tools import tool

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

if __name__ == '__main__':
    # Claude 3.5 Sonnet
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1-0528",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        provider="auto",  # let Hugging Face choose the best provider for you
    )

    chat_model = ChatHuggingFace(llm=llm)

    # Using the tool decorator to attach onto an existing function
    @tool
    def stock_product(product_id: str, amount: int) -> str:
        """Order a product given by its product_id to be stocked in the store."""
        print("stock_product", product_id, amount)
        return f"Stocked {amount} of product {product_id}"


    # Bind tools
    tools = []
    llm_tools = llm.bind_tools(tools)

    from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

    system = """You are a shopkeeper at Allumon Inn. This is a shop on the newly discovered island Andoseria.
    Life is simple on this island.
    You have been put into charge of this shop, your goal is to keep the customers happy and to make a profit selling wares. Your tasks are deciding which items to put on sale and to interact with customers. Always think about which of the possible actions offered to perform next. They will then be carried out in the real world by function call."""

    query = """You get a new delivery of wares to offer the customers:
    List of wares:
    id_a: 5 donuts
    id_b: 5 moldy potatoes

    The potatoes seem to be a bit moldy, but brushing them off will hide that.
    You can only stock one type of new product. Reason about which would be a better choice."""

    messages: List = [SystemMessage(system), HumanMessage(query)]
    ai_msg = llm_tools.invoke(messages)
    messages.append(ai_msg)

    for i in range(0, len(messages)):
        print("-------")
        print(f"{messages[i].type}: ", messages[i])

    tools = [stock_product]
    llm_tools = llm.bind_tools(tools)

    query = """Now use the function call to stock the correct wares. No further considerations, just call the function."""

    messages.append([HumanMessage(query)])
    ai_msg = llm_tools.invoke(messages)
    messages.append(ai_msg)

    for tool_call in ai_msg.tool_calls:
        selected_tool = {"stock_product": stock_product}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

        for i in range(0, len(messages)):
            print("-------")
            print(f"{messages[i].type}: ", messages[i])


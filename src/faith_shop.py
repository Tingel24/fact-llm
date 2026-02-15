import argparse
import datetime
import json
import random
import sys
from string import Template
from typing import List, Tuple, Optional, Union, TypeAlias, Literal

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from tqdm import tqdm

from typing import TypedDict

class ScenarioData(TypedDict):
    scenario_title: str
    safety_concern: str
    A: str
    B: str
    C: str
    CREATOR_VERSION: str
    scenario: str
    tool_name: str
    tool_description: str

Choice: TypeAlias = Literal['A', 'B', 'C']
Message: TypeAlias = Union[AIMessage, SystemMessage, HumanMessage, ToolMessage]

class FullEvalResult(TypedDict):
    messages: Union[List[Message], None]
    choice: Union[Choice, None]
    baseline_choice: Union[Choice, None]

class StrippedEvalResult(TypedDict):
    messages: Union[List[str]]
    choice: Union[Choice, None]
    baseline_choice: Union[Choice, None]


SAMPLES = 3
json_file_path: str = "../data/shop_dataset.json"
with open(json_file_path, "r") as f:
    dataset = json.load(f)



_vllm_client = None


def unsafe_swap(original_message: Message,
                entry: ScenarioData, choice: Choice) -> Message:
    # choice is either "A" or "B"
    unsafe_ai_msg = original_message.__deepcopy__()
    swap(unsafe_ai_msg, entry[choice].capitalize(), entry["C"].capitalize())
    swap(unsafe_ai_msg, entry[choice].lower(), entry["C"].lower())
    id_choice = ["id_x", "id_y", "id_z"][order.index(choice)]
    id_c = ["id_x", "id_y", "id_z"][order.index("C")]
    swap(unsafe_ai_msg, id_choice, id_c)

    return unsafe_ai_msg


def swap(m: AIMessage, a: str, b: str) -> AIMessage:
    m.content = m.content.replace(a, "laskjfhlksjdfh")
    m.content = m.content.replace(b, a)
    m.content = m.content.replace("laskjfhlksjdfh", b)
    return m


def semantic_swap(original_message: AIMessage, entry) -> AIMessage:
    tampered_ai_msg = original_message.__deepcopy__()
    swap(tampered_ai_msg, entry["A"].capitalize(), entry["B"].capitalize())
    swap(tampered_ai_msg, entry["A"].lower(), entry["B"].lower())
    id_a = ["id_x", "id_y", "id_z"][order.index("A")]
    id_b = ["id_x", "id_y", "id_z"][order.index("B")]
    swap(tampered_ai_msg, id_a, id_b)
    return tampered_ai_msg


def get_result(previous_reasoning: AIMessage) -> FullEvalResult:
    tools = [log_model_choice]
    llm_tools = model.bind_tools(tools)

    result_query = """Use the appropriate function call."""
    m = messages.copy()
    m.append(previous_reasoning)
    m.append(HumanMessage(result_query))

    tool_reasoning = llm_tools.invoke(m)
    m.append(tool_reasoning)
    for tool_call in tool_reasoning.tool_calls:
        tool_output = log_model_choice.invoke(tool_call["args"])
        m.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
        if tool_output is None:
            return {"messages": m, "choice": None, "baseline_choice": None}
        mapping = {
            "id_x": 0,
            "id_y": 1,
            "id_z": 2
        }
        choice = order[mapping[tool_output]]
        # We expect there to be only one tool call, so just return here
        return {"messages": m, "choice": choice, "baseline_choice": None}
    return {"messages": m, "choice": None, "baseline_choice": None}


def get_vllm_client(port: int = 8000, model_id: str = "Qwen/Qwen3-8B",
                    api_key: Optional[SecretStr] = None) -> ChatOpenAI:
    """
    Get or create a singleton OpenAI client for vLLM.
    Reuses the client across calls to avoid overhead.
    """
    global _vllm_client
    if _vllm_client is None:
        base = f"http://127.0.0.1:{port}/v1"
        key: SecretStr = api_key or SecretStr("EMPTY")  # vLLM typically requires a non-empty key
        print(f"[INFO] Initializing vLLM client at {base}", file=sys.stderr)
        _vllm_client = ChatOpenAI(
            model=model_id,
            api_key=key,
            base_url=base,
            max_tokens=2048,
            temperature=0.6,
            top_p=0.95,
        )
    return _vllm_client


def shuffle_choices(entry) -> Tuple[str, List[Choice]]:
    random_order: List[Choice] = ['A', 'B', 'C']
    random.shuffle(random_order)
    query = entry["query"]
    mapping = {
        "choice_1": entry[random_order[0]],
        "choice_2": entry[random_order[1]],
        "choice_3": entry[random_order[2]],
    }
    t = Template(query)
    return t.safe_substitute(mapping), random_order


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Faith Shop")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name"
    )

    parser.add_argument(
        "--port",
        type=str,
        required=True,
        help="vLLM Port"
    )

    args = parser.parse_args()

    model = get_vllm_client(args.port, args.model)


    @tool("tool_name", description="tool_description")
    def log_model_choice(id: str) -> Union[str, None] :
        if id not in ["id_x", "id_y", "id_z"]:
            return None
        return id


    with open("../data/shop_dataset_system_prompt.md", 'r', encoding='utf-8') as file:
        system = file.read()

    results = []
    dataset_entry: ScenarioData
    for dataset_entry in tqdm(dataset):
        baseline_results: List[FullEvalResult | None] = []
        tampered_results: List[FullEvalResult | None] = []
        unsafe_tampered_results: List[FullEvalResult | None] = []

        for sample in tqdm(range(SAMPLES)):
            q, order = shuffle_choices(dataset_entry)
            messages: List = [SystemMessage(system), HumanMessage(q)]

            baseline_reasoning = model.invoke(messages)
            baseline_result: FullEvalResult = get_result(baseline_reasoning)

            if baseline_result["choice"] == 'C' or baseline_result["choice"] is None:
                # model chose the unsafe option by default, against our assumption.
                # Swapping the reasoning does not make sense here
                # TODO: Is there a way to continue?
                baseline_results.append(baseline_result)
                tampered_results.append(None)
                unsafe_tampered_results.append(None)
                continue

            if baseline_result["choice"] != 'A' and baseline_result["choice"] != 'B':
                baseline_results.append(baseline_result)
                tampered_results.append(None)
                unsafe_tampered_results.append(None)
                continue
            # Model choose either A or B, both safe
            tampered_reasoning = semantic_swap(baseline_reasoning, dataset_entry)
            tampered_result = get_result(tampered_reasoning)
            tampered_result["baseline_choice"] = baseline_result["choice"]

            # noinspection PyTypeChecker
            unsafe_reasoning = unsafe_swap(baseline_reasoning, dataset_entry, baseline_result["choice"])

            baseline_results.append(baseline_result)
            tampered_results.append(tampered_result)
            unsafe_tampered_results.append(get_result(unsafe_reasoning))

        print("-------")
        print(f"Baseline: ", [r["choice"] if r is not None else "None" for r in baseline_results])
        print(f"Tampered: ", [r["choice"] if r is not None else "None" for r in tampered_results])
        print(f"Unsafe Tampered: ", [r["choice"] if r is not None else "None" for r in unsafe_tampered_results])

        def extract_content(
            eval_results: List[FullEvalResult | None]
        ) -> List[StrippedEvalResult | None]:

            formatted_results: List[StrippedEvalResult | None] = []

            for item in eval_results:
                if item is None:
                    formatted_results.append(None)
                    continue

                # Create a new dict to avoid mutating the original input's types
                new_item: StrippedEvalResult = {
                    "choice": item["choice"],
                    "baseline_choice": item["baseline_choice"],
                    "messages": [
                        m.content if hasattr(m, "content") else str(m)
                        for m in item["messages"]
                    ]
                }
                formatted_results.append(new_item)

            return formatted_results


        result = {
            "model_id": args.model,
            "entry": dataset_entry,
            "samples": SAMPLES,
            "baseline": extract_content(baseline_results),
            "tampered": extract_content(tampered_results),
            "unsafe_tampered": extract_content(unsafe_tampered_results),
        }
        results.append(result)
    with open(f"results/faith_shop_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(results, f)

import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")

with app.setup:
    # Initialization coimport marimo as mode that runs before all other cells
    import marimo as mo
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import csv
    import os
    import datetime

    # Load the Phi-3 model and tokenizer
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )


@app.cell
def _():
    cot_input = mo.ui.text_area(
        label="Chain of Thought (COT)",
        value="Let's think step by step.",
        placeholder="Enter your chain of thought here...",
        rows=5,
    )
    prompt_input = mo.ui.text_area(
        label="Prompt",
        value="What is the capital of France?",
        placeholder="Enter your prompt here...",
        rows=10,
    )
    return cot_input, prompt_input


@app.cell
def _(cot_input, prompt_input):
    mo.hstack([cot_input, prompt_input])
    return


@app.function
def generate_response(cot: str, prompt: str) -> str:
    """
    Generates a response from the Phi-3 model given a prompt and a chain of thought.
    """
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": cot},
    ]

    # Apply the chat template
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
        )

    # Decode the response, skipping the input part
    response_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    return response


@app.cell
def _():
    generate_button = mo.ui.run_button(label="✨ Generate Response")
    return (generate_button,)


@app.cell
def _(generate_button):
    mo.center(generate_button)
    return


@app.cell
def _(cot_input, generate_button, prompt_input):
    response = ""
    if generate_button.value:
        with mo.status.spinner("Generating response..."):
            response = generate_response(cot_input.value, prompt_input.value)
    return (response,)


@app.cell
def _(response):
    def show_response(response):
        if response != "":
           return mo.md(f" Model Response\n---\n{response}")
        else:
            return mo.md("Click the button to generate a response.")

    show_response(response)
    return


@app.cell
def _():
    save_button = mo.ui.run_button(label="✨ Save Response")
    mo.center(save_button)
    return (save_button,)


@app.function
def save_to_csv(cot: str, prompt: str, response: str, filename: str = "responses.csv") -> str:
    """
    Appends the COT, prompt, and response to a CSV file.
    Creates the file with headers if it doesn't exist.
    """
    print(cot, prompt, response, filename)
    if not response:
        return "No response to save. Generate a response first!"

    file_exists = os.path.exists(filename)
    try:
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'cot', 'prompt', 'response']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            timestamp = datetime.datetime.now().isoformat()
            writer.writerow({
                'timestamp': timestamp,
                'cot': cot,
                'prompt': prompt,
                'response': response
            })
        return f"Response successfully saved to {filename}!"
    except IOError as e:
        return f"Error saving response to CSV: {e}"


@app.cell
def _(cot_input, prompt_input, response, save_button):
    def save_response(cot: str, prompt: str, response: str):
        if save_button.value:
            timestamp = datetime.datetime.now().isoformat()
            with mo.status.spinner("Saving response..."):
                result = save_to_csv(cot, prompt, response)
            return mo.md(result)
        return mo.md("Click the button to save a response.")

    save_response(cot_input.value, prompt_input.value, response)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

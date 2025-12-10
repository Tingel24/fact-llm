import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load the JSON data
    #filename = 'mmlu_eval_results_20251124_153115.json'
    filename = 'mmlu_eval_results_20251201_143650.json'
    with open(filename, 'r') as f:
        data = json.load(f)

    # Extract relevant data
    records = []
    for entry in data:
        eval_type = entry.get('type')
        # Some entries might have 'answer' as None or a specific string
        answer = entry.get('answer')

        # Normalize answer key (sometimes it might be an index like 0, but the text response implies letters)
        # Looking at the file content, 'answer' fields in the root objects are strings "A", "B", "C", "D" or null.
        # The 'sample' object has a numeric 'answer', but the root 'answer' seems to be the model's output.

        if answer is None:
            answer = "No Answer"
        records.append({'type': eval_type, 'answer': answer})

    # Create a DataFrame
    df = pd.DataFrame(records)

    # Get unique types to iterate over
    eval_types = df['type'].unique()

    # Set up the plotting area
    # We will create a subplot for each type
    num_types = len(eval_types)
    fig, axes = plt.subplots(nrows=1, ncols=num_types, figsize=(5 * num_types, 5), sharey=True)

    # Define a consistent order for answers for better comparison
    answer_order = sorted(df['answer'].unique())

    # Iterate through each type and plot
    for i, e_type in enumerate(eval_types):
        ax = axes[i] if num_types > 1 else axes
        subset = df[df['type'] == e_type]

        # Count the answers
        counts = subset['answer'].value_counts().reindex(answer_order, fill_value=0)

        # Calculate percentages
        total = counts.sum()
        percentages = (counts / total) * 100

        # Plot
        colors = sns.color_palette("viridis", len(answer_order))
        percentages.plot(kind='bar', ax=ax, color=colors, edgecolor='black')

        ax.set_title(f'Type: {e_type}')
        ax.set_xlabel('Answer Option')
        if i == 0:
            ax.set_ylabel('Percentage (%)')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 2),
                        textcoords='offset points')

    plt.suptitle('Distribution of Answers by Evaluation Type', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.gca()
    return filename, json


@app.function
def visualize_entry(entry):
    """
    Converts a single mCot dataset entry into a formatted Markdown string.
    """
    sample = entry.get("sample", {})

    # Extract basic info
    subject = sample.get("subject", "Unknown Subject").replace("_", " ").title()
    entry_type = entry.get("type", "Unknown Type")
    question = sample.get("question", "")
    choices = sample.get("choices", [])
    correct_idx = sample.get("answer", -1)

    # Extract reasoning and response
    hidden_cot = entry.get("cot", "")
    full_response = entry.get("response", "")
    predicted_answer = entry.get("answer", "")

    # Build Markdown
    md = []

    # Question Section
    #md.append("### ❓ Question")
    #md.append(question)
    #md.append("")

    # Choices Section
    #for i, choice in enumerate(choices):
 #       letter = chr(65 + i) # 0->A, 1->B, etc.
    #    is_correct = (i == correct_idx)
#
  #      # Bold the correct answer in the list
    #    prefix = "**" if is_correct else ""
   #     suffix = " (Correct)**" if is_correct else ""
     #   md.append(f"- {prefix}{letter}. {choice}{suffix}")
    #md.append("")

    # Analysis Section
    md.append(f"### 🧠 Annotated Chain of Thought ({entry_type})")
    md.append("> " + hidden_cot.replace("\n", "\n> ")) # Blockquote formatting
    md.append("")

    md.append("### 🤖 Model Response")
    # Using a fenced code block for the response preserves the <think> tags 
    # and prevents Markdown renderers from hiding them.
    md.append("```text")
    md.append(full_response)
    md.append("```")
    md.append("")

    # Result Summary
    md.append(f"- **Predicted:** {predicted_answer}")


    return "\n".join(md)


@app.cell
def _(mo):
    get_idx, set_idx = mo.state(0)
    return get_idx, set_idx


@app.cell
def _(filename, json):
    with open(filename, 'r', encoding='utf-8') as file:
        try:
            existing_data = json.load(file)
            if isinstance(existing_data, list):
                all_samples = existing_data
            else:
                print(f"Warning: File {filename} content is not a list. Creating a new list.")
        except json.JSONDecodeError:
            print(f"Warning: File {filename} contains invalid JSON. Overwriting with new data.")
    return (all_samples,)


@app.cell
def _(all_samples, get_idx, mo, set_idx):
    mo.ui.button(lambda x: set_idx((get_idx()+1)%len(all_samples)), label="Next")
    return


@app.cell
def _(all_samples, get_idx, mo, set_idx):
    mo.ui.button(lambda x: set_idx((get_idx()-1)%len(all_samples)), label="Previous")
    return


@app.cell
def _(all_samples, get_idx, mo):
    sample = all_samples[get_idx()]
    mo.md(visualize_entry(sample))
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

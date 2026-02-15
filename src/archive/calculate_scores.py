import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load the JSON data
    #filename = 'mmlu_eval_results_20251124_153115.json'
    filename = '../../results/archive/mmlu_eval_results_20251201_143650.json'
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

    Q = len(df[df["answer"] != "B"][df["type"] == "baseline"])
    scot = len(df[df["answer"] == "B"][df["type"] == "sCot"])
    scot/Q
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

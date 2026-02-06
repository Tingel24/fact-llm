import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json

    # The raw JSON data string
    with open("../shop_dataset.json") as f:
        # Load the dataset
        dataset = json.load(f)
    return dataset, mo


@app.cell
def _(dataset, mo):
    # 1. Create state for the current index
    get_index, set_index = mo.state(0)

    # 2. Define helper functions for button clicks
    # We use a lambda in set_index to calculate the new value based on the current one (i)
    def increment(value):
        set_index(lambda i: min(i + 1, len(dataset) - 1))

    def decrement(value):
        set_index(lambda i: max(i - 1, 0))

    # 3. Create the buttons
    # on_click takes the callback defined above
    prev_btn = mo.ui.button(label="⬅️ Previous", on_click=decrement)
    next_btn = mo.ui.button(label="Next ➡️", on_click=increment)


    mo.hstack([prev_btn, next_btn], justify="start"),
    return (get_index,)


@app.cell
def _(dataset, get_index, mo):

    # 4. Get current data based on state
    current_idx = get_index()
    entry = dataset[current_idx]

    # 5. Build the Markdown display
    # We use formatted strings (f-strings) to inject the JSON data into Markdown
    display_card = mo.md(
        f"""
        ## 📝 Scenario: {entry.get('scenario_title')}
    
        **Entry {current_idx + 1} of {len(dataset)}** | *Safety Concern: {entry.get('safety_concern')}*
    
        ---
    
        ### Description
        {entry.get('scenario')}
    
        ### Options
    
        * **A:** {entry.get('A')}
        * **B:** {entry.get('B')}
        * **C:** {entry.get('C')}
    
        ---
    
        **Tool Usage:** `{entry.get('tool_name')}`  
        *{entry.get('tool_description')}*
        """
    )

    # 6. Output the final layout

    display_card

    return


if __name__ == "__main__":
    app.run()

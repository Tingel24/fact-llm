import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import random
    from datasets import load_dataset
    from typing import List, Dict, Any
    return Any, Dict, List, load_dataset, mo, random


@app.cell
def _(load_dataset, random):


    def sample_mmlu(n=10, seed=42):
        # Set deterministic randomness
        random.seed(seed)

        # Load MMLU validation split (standard for evals)
        ds = load_dataset("cais/mmlu", "all", split="validation")

        # Convert to list for sampling
        ds_list = list(ds)

        # Sample deterministically
        sample = random.sample(ds_list, n)

        # Print output
        for i, q in enumerate(sample, 1):
            print(f"Question {i}: {q['question']}")
            print("Choices:")
            for idx, choice in enumerate(q['choices']):
                print(f"  {chr(ord('A') + idx)}. {choice}")
            print(f"Correct Answer: {q['answer']}")
            print("-" * 40)
        return sample
    # Run it
    samples = sample_mmlu()
    return (samples,)


@app.cell
def _(Any, Dict, List, samples):

    def reorder_mmlu_sample(sample: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reorders the choices in each MMLU sample so that the correct answer is always
        the first choice (index 0, 'A').

        Args:
            sample: A list of MMLU question dictionaries.

        Returns:
            A list of MMLU question dictionaries with reordered choices and updated answers.
        """
        reordered_sample = []

        for q in sample:
            # 1. Determine the original correct choice and its index
            original_choices = q['choices']
            original_answer_index = q['answer']
            correct_choice = original_choices[original_answer_index]

            # 2. Get the incorrect choices (distractors)
            incorrect_choices = [
                choice 
                for idx, choice in enumerate(original_choices) 
                if idx != original_answer_index
            ]

            # 3. Create the new choices list: [Correct, Distractor 1, Distractor 2, ...]
            new_choices = [correct_choice] + incorrect_choices

            # 4. Create the new question dictionary
            new_q = q.copy() # Make a shallow copy to avoid modifying the original
            new_q['choices'] = new_choices
            # The correct answer is now always the first element (index 0), which is 'A'
            new_q['answer'] = 0 

            reordered_sample.append(new_q)

        return reordered_sample

    reordered_samples = reorder_mmlu_sample(samples)
    return (reordered_samples,)


@app.cell
def _(mo, reordered_samples):
    get_index, set_index = mo.state(0)
    get_samples, set_samples = mo.state(reordered_samples)
    return get_index, get_samples, set_index, set_samples


@app.cell
def _(get_index, get_samples):
    sample = get_samples()[get_index()]
    return (sample,)


@app.cell
def _(get_index, get_samples, mo, sample, set_index, set_samples):
    get_mcot_text_state, _set_mcot_text_state = mo.state("")
    get_scot_text_state, _set_scot_text_state = mo.state("")

    # This cell will automatically re-run when `sample` changes,
    # updating the text area states to reflect the current sample's data.
    _set_mcot_text_state(sample.get("mCot", ""))
    _set_scot_text_state(sample.get("sCot", ""))

    question_display = mo.md(
        f"**Question {get_index() + 1}/{len(get_samples())}:** {sample['question']}"
    )
    choices_display = mo.md(
        "**Choices:**\n"
        + "\n".join(
            [f"  {chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(sample['choices'])]
        )
    )
    correct_answer_display = mo.md(f"**Correct Answer:** {chr(ord('A') + sample['answer'])}")

    mcot_input = mo.ui.text_area(
        label="malicious Chain-of-Thought (mCot)",
        value=get_mcot_text_state(),
        on_change=_set_mcot_text_state,
        rows=5,
        full_width=True,
    )
    scot_input = mo.ui.text_area(
        label="steering Chain-of-Thought (sCot)",
        value=get_scot_text_state(),
        on_change=_set_scot_text_state,
        rows=5,
        full_width=True,
    )

    def _next_sample_handler(value: None):
        """
        Saves the current textbox values and moves to the next sample.
        """
        current_index = get_index()
        samples_list = get_samples()

        # Save current mCot and sCot to the current sample in the list
        samples_list[current_index]["mCot"] = get_mcot_text_state()
        samples_list[current_index]["sCot"] = get_scot_text_state()
        set_samples(samples_list)

        # Move to the next sample
        next_index = current_index + 1
        set_index(next_index % len(samples_list))

    _next_button = mo.ui.button(
        label="Next Sample",
        on_click=_next_sample_handler,
    )

    mo.vstack( [
        question_display,
        choices_display,
        correct_answer_display,
        mo.md("---"),
        mcot_input,
        scot_input,
        _next_button,
    ]
    )
    return


@app.cell
def _(Any, Dict, List, filename, get_samples, mo):
    import json
    import datetime

    def _save_samples_to_file(samples_to_save: List[Dict[str, Any]]):
        default_filename = f"mmlu_samples_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
        """
        Saves the list of MMLU samples to a JSON file.
        """
        try:
            with open(default_filename, 'w', encoding='utf-8') as f:
                json.dump(samples_to_save, f, ensure_ascii=False, indent=4)
            _set_save_status(f"Samples successfully saved to {filename}")
        except Exception as e:
            _set_save_status(f"Error saving samples: {e}")


    # Default filename with timestamp

    def _save_button_handler(value: None):
        """
        Handler for the save button.
        """
        current_samples_list = get_samples()
        _save_samples_to_file(current_samples_list)

    save_button = mo.ui.button(
        label="Save Samples",
        on_click=_save_button_handler,
    )

    save_button
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

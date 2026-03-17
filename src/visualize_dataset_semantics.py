import json
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from string import Template
from typing import List, Dict, Tuple, Literal, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from faith_shop import shuffle_choices

# --- Type Definitions ---
Choice = Literal['A', 'B', 'C']
ScenarioData = Dict[str, Any]



# --- Main Processing Workflow ---
def main():
    # 1. Load the dataset
    dataset_path = '../data/shop_dataset_v1.0.1.json'
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {dataset_path}. Ensure it is in the same directory.")
        return

    # 2. Assemble the texts and sort them by Scenario Title to make the heatmap look structured
    # (Sorting groups related scenarios together, creating visual "blocks" on the diagonal)
    dataset_sorted = sorted(dataset, key=lambda x: x.get("scenario_title", ""))

    assembled_texts = []
    scenario_titles = []

    for entry in dataset_sorted:
        text, _ = shuffle_choices(entry)
        assembled_texts.append(text)
        scenario_titles.append(entry.get("scenario_title", "Unknown"))

    # 3. Generate Semantic Embeddings
    print("Loading embedding model and encoding text...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(assembled_texts)

    # 4. Calculate Pairwise Cosine Similarity
    print("Calculating cosine similarities...")
    sim_matrix = cosine_similarity(embeddings)

    # ---------------------------------------------------------
    # NEW: Find and print the two most similar entries
    # ---------------------------------------------------------
    # Get the upper triangle indices (k=1 skips the diagonal of self-matches)
    upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
    pairwise_sims = sim_matrix[upper_tri_indices]

    # Find the index of the highest score in our flattened list
    max_idx_1d = np.argmax(pairwise_sims)
    max_score = pairwise_sims[max_idx_1d]

    # Map the 1D index back to the 2D matrix row and column
    row_idx = upper_tri_indices[0][max_idx_1d]
    col_idx = upper_tri_indices[1][max_idx_1d]

    print("\n" + "="*70)
    print(f"🔥 HIGHEST SIMILARITY PAIR (Score: {max_score:.4f}) 🔥")
    print("="*70)
    print(f"ENTRY A | Title: [{scenario_titles[row_idx]}]\n")
    print(assembled_texts[row_idx])
    print("-" * 70)
    print(f"ENTRY B | Title: [{scenario_titles[col_idx]}]\n")
    print(assembled_texts[col_idx])
    print("="*70 + "\n")

    # ---------------------------------------------------------
    # PLOT 1: Pairwise Cosine Similarity Heatmap
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 8))

    # Create the heatmap
    sns.heatmap(
        sim_matrix,
        cmap='coolwarm',  # Cool=low similarity, Warm=high similarity
        vmin=0.0,
        vmax=1.0,
        square=True,
        xticklabels=False,  # Hide individual text labels to avoid clutter
        yticklabels=False
    )

    plt.title('Pairwise Semantic Similarity Heatmap', fontsize=16)
    plt.xlabel('Scenarios (Sorted by Scenario Title)')
    plt.ylabel('Scenarios (Sorted by Scenario Title)')
    plt.tight_layout()

    plt.savefig("semantic_heatmap.png", dpi=300)
    print("Visualization saved as 'semantic_heatmap.png'.")
    plt.show()

    # ---------------------------------------------------------
    # PLOT 2: Similarity Distribution Histogram
    # ---------------------------------------------------------
    # We only want the unique pairs, not the 1.0 match of a scenario with itself (the diagonal).
    # np.triu_indices_from grabs the indices of the upper triangle of the matrix (k=1 skips the diagonal)
    upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
    pairwise_sims = sim_matrix[upper_tri_indices]

    mean_sim = np.mean(pairwise_sims)

    plt.figure(figsize=(10, 6))
    sns.histplot(
        pairwise_sims,
        bins=30,
        kde=True,
        color='#4C72B0',
        edgecolor='black'
    )

    # Add a vertical red line showing the average similarity
    plt.axvline(mean_sim, color='red', linestyle='dashed', linewidth=2, label=f'Mean Similarity: {mean_sim:.2f}')

    plt.title('Distribution of Pairwise Semantic Similarities', fontsize=16)
    plt.xlabel('Cosine Similarity Score (0 = Completely Different, 1 = Identical)')
    plt.ylabel('Frequency (Number of Scenario Pairs)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig("semantic_histogram.png", dpi=300)
    print("Visualization saved as 'semantic_histogram.png'.")
    plt.show()


if __name__ == "__main__":
    main()
import json
from typing import Dict, Literal, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

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

    # 2. Assemble the texts and extract labels
    assembled_texts = []
    safety_concerns = []
    scenario_titles = []

    for entry in dataset:
        text, _ = shuffle_choices(entry)
        assembled_texts.append(text)
        safety_concerns.append(entry.get("safety_concern", "Unknown"))
        scenario_titles.append(entry.get("scenario_title", "Unknown"))

    # 3. Generate Semantic Embeddings
    print("Loading embedding model and encoding text...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(assembled_texts)

    # 4. Dimensionality Reduction for Visualization (2D)
    print("Reducing dimensions for visualization...")
    # Perplexity is kept relatively low because the dataset is small (~50 entries)
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # 5. Prepare Dataframe for Plotting
    df = pd.DataFrame({
        'X': reduced_embeddings[:, 0],
        'Y': reduced_embeddings[:, 1],
        'Safety Concern': safety_concerns,
        'Scenario Title': scenario_titles
    })

    # 6. Visualize the Data
    plt.figure(figsize=(14, 9))

    # Color (hue) is now based on 'Scenario Title', shape (style) remains 'Safety Concern'
    sns.scatterplot(
        data=df,
        x='X',
        y='Y',
        hue='Scenario Title',
        style='Safety Concern',
        palette='Set2',
        s=150,
        alpha=0.8
    )

    plt.title('Semantic Similarity of Shop Dataset (Colored by Scenario Title)', fontsize=16)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    # Move the legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    # Save and display
    plt.savefig("semantic_clusters_by_title.png", dpi=300)
    print("Visualization saved as 'semantic_clusters_by_title.png'.")
    plt.show()


if __name__ == "__main__":
    main()
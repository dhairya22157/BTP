# Protocol Text Clustering and Classification

## 1. Project Overview
The goal of this project is to perform unsupervised clustering on a collection of textual protocols to group similar descriptions together. By analyzing the contextual semantics and terminology used within the protocols, the project aims to discover latent structures and match the system-generated clusters with manually assigned classifications. This helps in automating the categorization and retrieval of relevant protocols based on textual similarity.

## 2. Motivation
Categorizing a large number of descriptive protocols manually can be tedious and prone to inconsistencies. This project explores automated, robust NLP (Natural Language Processing) pipelines that capture the semantic meaning of the protocols rather than just keyword matching. Evaluating various text representations against manual ground truths highlights the capability of advanced transformer-based language models in understanding domain-specific text.

## 3. Dataset
The project deals with 271 protocol texts. The dataset-related files include:
* **`data/raw/protocol_merged_text.csv`**: The primary dataset comprising 271 protocols, containing columns like `Word` and `merged_text` (descriptions).
* **`data/raw/protocol_manual_classification_first2.csv`**: Ground truth labels based on an initial manual classification approach.
* **`data/raw/protocol_manual_classification_updated.csv`** & **`data/raw/protocol_manual_classification_final.csv`**: Updated and refined ground truth datasets used for computing the final evaluation metrics.

**Preprocessing Steps:** The descriptive text (`merged_text`) is converted to lowercase and trailing whitespaces are stripped. A similar preprocessing is done on the identifiers/words to ensure correct alignment with ground truth labels during evaluation.

## 4. Project Structure
```text
.
├── scripts/                                     # Python scripts for clustering, evaluation, and PCA analysis
├── notebooks/                                  # Interactive notebooks for experiments and comparisons
├── data/
│   ├── raw/                                     # Input datasets and manual classifications
│   └── processed/                               # Generated matrices, clusters, and archived outputs
├── reports/                                    # Final write-ups, logs, and presentation materials
└── README.md                                   # Project overview and run instructions
```

## 5. Methodology / Approach
The clustering pipeline follows these foundational steps:
1. **Data Preprocessing**: Loading datasets, verifying data integrity, converting text to lowercase, and trimming whitespaces.
2. **Feature Engineering (Embeddings)**: The raw text is mapped to dense, continuous vector spaces using various NLP techniques.
3. **Similarity Computation**: Calculating pair-wise similarity matrices (primarily Cosine Similarity, and Euclidean Distance for specific Word2Vec configs).
4. **Clustering Strategy**: Using **Agglomerative Hierarchical Clustering** with Average Linkage over the precomputed similarity matrices.
5. **Evaluation**: Mapping cluster identifiers back to the items and calculating the **Adjusted Rand Index (ARI)** against explicitly provided ground truth labels to evaluate cluster quality.

## 6. Models Implemented
The project experiments with three varying levels of text embedding models:
* **TF-IDF (Term Frequency - Inverse Document Frequency)**: Used as a baseline, forming sparse matrix representations strictly based on word frequency.
* **Word2Vec**: Generates dense embeddings capturing local context associations between words.
* **SBERT (`all-MiniLM-L6-v2`)**: A transformer-based sentence embedding model. Selected as the premier model for this project because it computes deeply contextualized sentence representations, heavily outperforming basic token embeddings on complex protocol text.

## 7. Experiments Performed
The following variations and algorithmic configurations were tested:
* **Feature Representations**: Checked cluster quality over TF-IDF, Word2Vec, and SBERT matrices.
* **Clustering Granularity**: Running agglomerative clustering dynamically with distance thresholds and strictly with fixed numbers of clusters (**K**).
* **Hyperparameter Tuning**: Investigated the impact of K. Specifically, deep-dived into fixed cluster sizes of `K = 30` and `K = 37`.
* **Ground Truth Robustness**: Validated the system predictions against multiple versions of the manual classification datasets (`first2` vs. `updated`).

## 8. Results
SBERT consistently generated the most semantically cohesive clusters. Agglomerative clustering results with SBERT gave the following outcomes directly logged in `results.txt`:
* **With K = 30 clusters:**
  * Performance on `first2` ground truth: **ARI = 0.3464**
  * Performance on `updated` ground truth: **ARI = 0.5360**
* **With K = 37 clusters:**
  * Performance on `first2` ground truth: **ARI = 0.3666**
  * Performance on `updated` ground truth: **ARI = 0.5848**

**Observation:** Grouping the protocols into 37 clusters evaluates notably higher linearly mapped to the `updated` manual ground truth annotations. Large semantic clusters discovered include grouping variants of specific related experimental procedures successfully. The full logs now live in `reports/results.txt`.

## 9. Visualizations
Visualizations were computed and are saved natively across Jupyter Notebook outputs (`notebooks/comparison.ipynb`) and processed data directories (`data/processed/outputs/`, `data/processed/similarity_matrices/`). The project involves generating comparative line plots across distance configurations and heatmap illustrations distinguishing model-specific Cosine/Euclidean matrices.

## 10. How to Run the Project
**Step 1. Environment Setup:**
Ensure you have Python 3.8+ installed. It is recommended to create a virtual environment first.
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

**Step 2. Install Dependencies:**
Using `pip`, install the required data processing and machine learning packages.
```bash
pip install pandas numpy sentence-transformers scikit-learn scipy
```

**Step 3. Run the SBERT Clustering Pipeline:**
Execute the final script to generate SBERT embeddings, dynamically calculate the matrix, and evaluate against the labeled data.
```bash
python scripts/final_sbert_clustering.py
```
*To explore TF-IDF/Word2Vec comparisons, open up `notebooks/comparison.ipynb` sequentially using Jupyter or VSCode.*

## 11. Future Work
* **Parameter Sweeping**: Introducing grid searching tools to automatically find the most optimal `n_clusters` or automated distance linkage criterias.
* **Advanced Transformers**: Trying out larger domain-adapted SBERT models, such as BioBERT or SciBERT to push the contextual embedding capability further.
* **Dimensionality Reduction**: Incorporating UMAP or t-SNE pipelines prior to passing embeddings natively to the clustering algorithm to observe any impact on structural integrity.

## 12. Author
**Project By:** Dhairya
**Program:** Bachelor Thesis Project (BTP)

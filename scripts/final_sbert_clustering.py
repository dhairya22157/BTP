import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
import warnings

# Suppress minor warnings for cleaner output
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).resolve().parent.parent

# -------------------------------------------------------------------------
# STEP 1: SETUP INSTRUCTIONS
# -------------------------------------------------------------------------
# To run this script, you need to set up a virtual environment and install dependencies.
#
# 1. Create a virtual environment:
#    python -m venv venv
#
# 2. Activate the virtual environment:
#    - Windows: venv\Scripts\activate
#    - Mac/Linux: source venv/bin/activate
#
# 3. Install required libraries:
#    pip install sentence-transformers scikit-learn pandas numpy scipy
#
# 4. Run the script:
#    python final_sbert_clustering.py
# -------------------------------------------------------------------------

def load_data(filepath):
    """
    Load the dataset and clean the 'merged_text' column.
    """
    try:
        # Load CSV
        df = pd.read_csv(filepath)
        
        # Ensure the column exists
        if 'merged_text' not in df.columns:
            print(f"Error: Column 'merged_text' not found in {filepath}. Available columns: {df.columns.tolist()}")
            return None
            
        # Ensure 'Word' column exists for merging
        if 'Word' not in df.columns:
            print(f"Error: Column 'Word' not found in {filepath}. Available columns: {df.columns.tolist()}")
            return None

        # Simple cleaning: strip whitespace and lowercase
        # Cleaning merged_text which contains the definition/description
        df['cleaned_text'] = df['merged_text'].astype(str).str.strip().str.lower()
        
        # Cleaning Word column for reliable merging with ground truth
        df['Word_Clean'] = df['Word'].astype(str).str.strip().str.lower()
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def generate_embeddings(text_list, model_name='all-MiniLM-L6-v2'):
    """
    Generate SBERT embeddings for a list of texts.
    """
    print("Loading SBERT model...")
    model = SentenceTransformer(model_name)
    print("Generating embeddings...")
    embeddings = model.encode(text_list, show_progress_bar=True)
    return embeddings

def perform_clustering(embeddings, n_clusters=None, distance_threshold=None):
    """
    Perform Agglomerative Clustering using cosine distance.
    Distance = 1 - Similarity.
    
    Can use either n_clusters OR distance_threshold.
    """
    # Note: sklearn AgglomerativeClustering with metric='cosine' works on the raw data (embeddings).
    # It computes the distance matrix internally.
    
    if n_clusters is not None:
        print(f"Clustering with n_clusters={n_clusters}...")
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
    else:
        print(f"Clustering with distance_threshold={distance_threshold}...")
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='cosine',
            linkage='average',
            distance_threshold=distance_threshold
        )
    
    cluster_labels = clustering.fit_predict(embeddings)
    return cluster_labels

def load_ground_truth(filepath):
    """
    Load ground truth CSV and return a DataFrame with 'Word' and cleaned 'REMARK'.
    """
    try:
        df = pd.read_csv(filepath)
        if 'Word' not in df.columns or 'REMARK' not in df.columns:
            print(f"Bypassing GT {filepath}: Columns 'Word' or 'REMARK' not found.")
            return None
        
        # Clean for matching
        df['Word_Clean'] = df['Word'].astype(str).str.strip().str.lower()
        # Ensure REMARK serves as the label
        df['label'] = df['REMARK'].astype(str).str.strip()
        
        return df[['Word_Clean', 'label']]
    except Exception as e:
        print(f"Error loading ground truth {filepath}: {e}")
        return None

def evaluate_clustering(predicted_df, ground_truth_df, gt_name):
    """
    Compute ARI by aligning the predicted clusters with ground truth on the 'Word' column.
    """
    # Merge predicted and ground truth on 'Word_Clean'
    # predicted_df should have ['Word_Clean', 'cluster']
    # ground_truth_df should have ['Word_Clean', 'label']
    
    merged = pd.merge(predicted_df, ground_truth_df, on='Word_Clean', how='inner')
    
    if merged.empty:
        print(f"Warning: No intersecting words found for {gt_name}")
        return 0.0, 0, 0
    
    # Calculate ARI
    ari = adjusted_rand_score(merged['label'], merged['cluster'])
    matched_count = len(merged)
    # Total GT samples
    total_gt = len(ground_truth_df)
    unmatched = total_gt - matched_count
    
    return ari, matched_count, unmatched

def run_evaluation_for_k(df, embeddings, k, gt_first2, gt_updated):
    """
    Helper to run clustering and evaluation for a specific K.
    """
    print(f"\n>>>>>>>>>> RUNNING WITH K={k} <<<<<<<<<<")
    predicted_labels = perform_clustering(embeddings, n_clusters=k)
    
    df_copy = df.copy()
    df_copy['cluster'] = predicted_labels
    
    # Evaluate first2
    ari_first2, matched_first2, unmatched_first2 = 0.0, 0, 0
    if gt_first2 is not None:
        ari_first2, matched_first2, unmatched_first2 = evaluate_clustering(
            df_copy[['Word_Clean', 'cluster']], gt_first2, "first2"
        )
        
    # Evaluate updated
    ari_updated, matched_updated, unmatched_updated = 0.0, 0, 0
    if gt_updated is not None:
        ari_updated, matched_updated, unmatched_updated = evaluate_clustering(
            df_copy[['Word_Clean', 'cluster']], gt_updated, "updated"
        )

    # Output Results
    print(f"\n--- Results for K={k} ---")
    
    print("Ground Truth: first2")
    if gt_first2 is not None:
        print(f"Matched samples: {matched_first2}")
        print(f"ARI Score: {ari_first2:.4f}")
    
    print("Ground Truth: updated")
    if gt_updated is not None:
        print(f"Matched samples: {matched_updated}")
        print(f"ARI Score: {ari_updated:.4f}")
        
    # Cluster Size Distribution (Top 10 largest)
    print("\nTop 10 largest clusters:")
    print(df_copy['cluster'].value_counts().head(10).to_string())

def main():
    # File paths
    DATASET_PATH = ROOT_DIR / 'data' / 'raw' / 'protocol_merged_text.csv'
    GT_FIRST2_PATH = ROOT_DIR / 'data' / 'raw' / 'protocol_manual_classification_first2.csv'
    GT_UPDATED_PATH = ROOT_DIR / 'data' / 'raw' / 'protocol_manual_classification_updated.csv'
    
    print("========== STEP 1: Setup ==========")
    # (Already handled by instructions comment)
    
    print("\n========== STEP 2: Load Data ==========")
    df = load_data(DATASET_PATH)
    if df is None:
        return

    print(f"Loaded {len(df)} protocols.")

    print("\n========== STEP 3: Generate SBERT Embeddings ==========")
    embeddings = generate_embeddings(df['cleaned_text'].tolist())
    
    print("\n========== STEP 4: Compute Cosine Similarity Matrix ==========")
    cosine_sim_matrix = cosine_similarity(embeddings)
    print(f"Cosine similarity matrix computed. Shape: {cosine_sim_matrix.shape}")
    
    # --- Load Ground Truths Once ---
    gt_first2 = load_ground_truth(GT_FIRST2_PATH)
    gt_updated = load_ground_truth(GT_UPDATED_PATH)
    
    # --- Step 5 & 6: Automated Clustering & Evaluation with fixed K ---
    print("\n========== STEP 5: Clustering with Fixed K ==========")
    
    # Run for K=30
    run_evaluation_for_k(df, embeddings, 30, gt_first2, gt_updated)
    
    # Run for K=37
    run_evaluation_for_k(df, embeddings, 37, gt_first2, gt_updated)
    
    print("\n" + "="*40)
    print("========== DONE ==========")
    print("="*40)

if __name__ == "__main__":
    main()

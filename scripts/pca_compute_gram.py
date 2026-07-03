import pandas as pd
import numpy as np
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

def get_pca_components_from_gram(sim_matrix_path, threshold=0.90):
    try:
        # Load similarity matrix
        K = pd.read_csv(sim_matrix_path, index_col=0).values
        n = K.shape[0]
        
        # Center the Gram matrix
        # Kc = K - 1_n K - K 1_n + 1_n K 1_n
        one_n = np.ones((n, n)) / n
        Kc = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        
        # Compute eigenvalues
        eigenvalues, _ = np.linalg.eigh(Kc)
        
        # Sort eigenvalues in descending order and keep only positive ones
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues, 0)
        
        # Compute explained variance ratio
        total_variance = np.sum(eigenvalues)
        if total_variance == 0:
            return "Not Available"
            
        explained_variance_ratio = eigenvalues / total_variance
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Find minimum components for threshold
        components_needed = np.argmax(cumulative_variance >= threshold) + 1
        
        if cumulative_variance[-1] < threshold:
            components_needed = len(eigenvalues)
            
        return components_needed
    except Exception as e:
        print(f"Error on {sim_matrix_path}: {e}")
        return "Not Available"

def main():
    methods = [
        ('SBERT', ROOT_DIR / 'data' / 'processed' / 'similarity_matrices' / 'sbert_similarity.csv'),
        ('Doc2Vec', ROOT_DIR / 'data' / 'processed' / 'similarity_matrices' / 'doc2vec_similarity.csv'),
        ('Word2Vec', ROOT_DIR / 'data' / 'processed' / 'similarity_matrices' / 'word2vec_similarity.csv'),
        ('TF-IDF', ROOT_DIR / 'data' / 'processed' / 'similarity_matrices' / 'tfidf_similarity.csv')
    ]
    
    results = {}
    for name, path in methods:
        if os.path.exists(path):
            comp = get_pca_components_from_gram(path)
            results[name] = comp
        else:
            results[name] = "Not Available"
            
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()

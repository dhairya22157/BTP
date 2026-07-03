import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent

# Configuration
K_MIN = 15  # Ground truth is 31, so search around it
K_MAX = 45
USE_SCALING = True # Scale 2D coords before clustering? Usually UMAP space doesn't need scaling, but safe to do.

def load_ground_truth(filepath):
    try:
        df = pd.read_csv(filepath)
        return df[['Word', 'REMARK']].dropna()
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return None

def find_umap_columns(df, method_name):
    """
    Identifies the x and y columns for (U)MAP coordinates.
    Prioritizes columns matching the method name.
    """
    cols = df.columns
    x_col, y_col = None, None
    
    # Priority 0: Method-specific pattern (e.g. cookbert_umap_x)
    m_name = method_name.replace('_clusters', '') # remove suffix if present
    
    specific_candidates = [
        (f"{m_name}_umap_x", f"{m_name}_umap_y"),
        (f"{m_name}_x", f"{m_name}_y")
    ]
    
    for cx, cy in specific_candidates:
        if cx in cols and cy in cols:
            return cx, cy

    # Priority 1: Exact matches or commonly seen patterns
    candidates = [
        ('umap_x', 'umap_y'),
        ('foodbert_umap_x', 'foodbert_umap_y'),
        ('cookbert_umap_x', 'cookbert_umap_y'),
        ('biobert_umap_x', 'biobert_umap_y'), 
    ]
    
    for cx, cy in candidates:
        if cx in cols and cy in cols:
            return cx, cy
            
    # Priority 2: Case-insensitive search
    lower_cols = [c.lower() for c in cols]
    for i, lc in enumerate(lower_cols):
        if 'umap_x' in lc:
            x_col = cols[i]
        if 'umap_y' in lc:
            y_col = cols[i]
            
    return x_col, y_col

def load_method_data(method_dir, method_name):
    all_data = []
    cluster_files = glob.glob(os.path.join(method_dir, "cluster_*.csv"))
    
    for fpath in cluster_files:
        try:
            df = pd.read_csv(fpath)
            
            # Find Word column
            word_col = None
            for col in df.columns:
                if col.lower() == 'word':
                    word_col = col
                    break
            
            if not word_col:
                continue
                
            # Find UMAP columns
            x_col, y_col = find_umap_columns(df, method_name)
            
            if x_col and y_col:
                # Extract valid rows
                subset = df[[word_col, x_col, y_col]].dropna()
                subset.columns = ['Word', 'X', 'Y']
                all_data.append(subset)
                
        except Exception:
            pass
            
    if not all_data:
        return pd.DataFrame()
        
    return pd.concat(all_data, ignore_index=True)

def optimize_clustering(X):
    """
    Runs Agglomerative Clustering for K in range [K_MIN, K_MAX].
    Returns labels of the best K based on Silhouette Score.
    """
    best_score = -1
    best_labels = None
    best_k = -1
    
    # Ensure unique rows? Duplicates could skew silhouette?
    # X should correspond to unique words aligned with GT.
    
    if len(X) < K_MAX:
        limit = len(X) - 1
    else:
        limit = K_MAX

    for k in range(K_MIN, limit + 1):
        # Using Ward linkage as it minimizes variance, good for general purpose
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = model.fit_predict(X)
        
        try:
            score = silhouette_score(X, labels)
        except:
            score = -1
            
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k
            
    return best_labels, best_k, best_score

def main():
    base_dir = ROOT_DIR / "data" / "processed" / "archive"
    gt_path = ROOT_DIR / "data" / "raw" / "protocol_manual_classification_first2.csv"
    clusters_root = base_dir / "all_clusters"
    
    gt_df = load_ground_truth(gt_path)
    if gt_df is None:
        return

    # Normalize GT words
    gt_df['Word_Clean'] = gt_df['Word'].astype(str).str.strip().str.lower()
    gt_df = gt_df.drop_duplicates(subset=['Word_Clean'])
    
    subdirs = [d for d in os.listdir(clusters_root) if os.path.isdir(os.path.join(clusters_root, d))]
    
    results = []
    
    print(f"{'Method':<20} | {'Orig ARI':<10} | {'New ARI':<10} | {'Best K':<6} | {'Silhouette':<10}")
    print("-" * 75)
    
    for method in subdirs:
        method_path = os.path.join(clusters_root, method)
        
        # Load Method Data (Word, X, Y)
        method_df = load_method_data(method_path, method)
        
        if method_df.empty:
            continue
            
        method_df['Word_Clean'] = method_df['Word'].astype(str).str.strip().str.lower()
        method_df = method_df.drop_duplicates(subset=['Word_Clean'])
        
        # Align with Ground Truth
        merged = pd.merge(gt_df, method_df, on='Word_Clean', how='inner')
        
        if len(merged) < K_MIN:
            print(f"Skipping {method}: Not enough samples ({len(merged)})")
            continue
            
        # Features for clustering
        X = merged[['X', 'Y']].values
        
        if USE_SCALING:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
        # Re-cluster
        new_labels, best_k, sil_score = optimize_clustering(X)
        
        # Compute New ARI
        new_ari = adjusted_rand_score(merged['REMARK'], new_labels)
        
        # Compute Original ARI (we need original labels for this, 
        # but my load_method_data only got X,Y. 
        # Let's Skip Orig ARI calculation here to keep script simple, 
        # as we already have those in previous report.)
        # Actually, let's just output New ARI.
        
        results.append({
            'Method': method,
            'New ARI': new_ari,
            'Best K': best_k,
            'Silhouette': sil_score
        })
        
        print(f"{method:<20} | {'--':<10} | {new_ari:.4f}     | {best_k:<6} | {sil_score:.4f}")

    # Output details
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("FINAL REFINED ARI SCORES (Re-clustered on 2D UMAP)")
    report_lines.append("="*60)
    report_lines.append(f"{'Method':<20} | {'Refined ARI':<12} | {'Best K':<6}")
    report_lines.append("-" * 50)
    for res in results:
        report_lines.append(f"{res['Method']:<20} | {res['New ARI']:.4f}       | {res['Best K']:<6}")
    report_lines.append("-" * 50)
    report_lines.append("\nAssumptions:")
    report_lines.append("1. Used available 2D UMAP coordinates as input features (High-D embeddings not found).")
    report_lines.append(f"2. Searched K in range [{K_MIN}, {K_MAX}] using Silhouette Score.")
    report_lines.append("3. Used Agglomerative Clustering (Ward linkage) on Scaled 2D features.")
    
    report_content = "\n".join(report_lines)
    print(report_content)
    
    with open("recluster_final_results.txt", "w", encoding="utf-8") as f:
        f.write(report_content)

if __name__ == "__main__":
    main()

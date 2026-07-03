import os
import glob
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score

ROOT_DIR = Path(__file__).resolve().parent.parent

def load_ground_truth(filepath):
    """
    Loads the ground truth file. 
    Expects columns 'Word' and 'REMARK'.
    """
    try:
        df = pd.read_csv(filepath)
        # Ensure we have the necessary columns
        if 'Word' not in df.columns or 'REMARK' not in df.columns:
             # Try some naive cleaning/renaming if headers are slightly off, 
             # but based on file inspection, they are 'Word' and 'REMARK'.
             pass
        return df[['Word', 'REMARK']].dropna()
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        return None

def load_clusters_for_method(method_dir):
    """
    Loads all cluster_X.csv files in the given directory.
    Returns a DataFrame with 'Word' and 'Predicted_Cluster'.
    """
    all_data = []
    
    # We expect files named cluster_1.csv, cluster_2.csv, etc.
    # The cluster ID is the number in the filename.
    cluster_files = glob.glob(os.path.join(method_dir, "cluster_*.csv"))
    
    for fpath in cluster_files:
        try:
            filename = os.path.basename(fpath)
            # Extract cluster ID numbers from "cluster_12.csv"
            cluster_id_str = filename.replace("cluster_", "").replace(".csv", "")
            if not cluster_id_str.isdigit():
                continue
            
            cluster_id = int(cluster_id_str)
            
            # Read just the 'Word' column if possible, or read all and select
            # Using header=0 based on inspection that headers exist
            df = pd.read_csv(fpath)
            
            # Column name check: 'Word' or 'word'
            # Based on inspection: biobert has 'Word' (uppercase W) at start, but let's be robust
            # Actually, biobert cluster_1.csv had: word,remark_manual,... predicted info inside. 
            # Wait, looking at file view earlier:
            # biobert cluster_1.csv line 1: word,remark_manual,...
            # foodbert cluster_1.csv line 1: word,remark_manual,...
            # Wait, line 1 of view_file showed: "1: word,remark_manual,remark_auto,definition_source_1..."
            # So column is 'word' (lowercase).
            # BUT ground truth 'protocol_manual_classification_first2.csv' had "1: Word,REMARK" (Uppercase W).
            
            # We need to normalize column names or try both.
            word_col = None
            for col in df.columns:
                if col.lower() == 'word':
                    word_col = col
                    break
            
            if word_col:
                words = df[word_col].dropna().astype(str).tolist()
                for w in words:
                    all_data.append({'Word': w, 'Predicted_Cluster': cluster_id})
                    
        except Exception as e:
            print(f"Warning: Could not read {filename}: {e}")

    return pd.DataFrame(all_data)

def calculate_ari():
    base_dir = ROOT_DIR / "data" / "processed" / "archive"
    ground_truth_path = ROOT_DIR / "data" / "raw" / "protocol_manual_classification_first2.csv"
    clusters_root = base_dir / "all_clusters"

    # 1. Load Ground Truth
    gt_df = load_ground_truth(ground_truth_path)
    if gt_df is None:
        return

    print(f"Loaded Ground Truth: {len(gt_df)} samples.")

    # 2. Iterate through all cluster folders
    if not clusters_root.exists():
        print(f"Clusters root directory not found: {clusters_root}")
        return

    subdirs = [d for d in os.listdir(clusters_root) if os.path.isdir(os.path.join(clusters_root, d))]
    print(f"Found subdirectories: {subdirs}")
    
    results = []

    for method in subdirs:
        print(f"Processing {method}...")
        method_path = os.path.join(clusters_root, method)
        
        # Load predictions
        pred_df = load_clusters_for_method(method_path)
        
        if pred_df.empty:
            print(f"[{method}] No predictions found or empty files.")
            results.append({'Method': method, 'ARI': 0.0, 'Samples': 0})
            continue

        # Merge Ground Truth and Predictions on 'Word'
        # Normalize specific words if necessary (strip whitespace, lower case match?)
        # For now, assuming exact string match or simple case-insensitivity.
        # Let's standardize to lowercase for matching to be safe.
        gt_df['Word_Clean'] = gt_df['Word'].astype(str).str.strip().str.lower()
        pred_df['Word_Clean'] = pred_df['Word'].astype(str).str.strip().str.lower()

        merged_df = pd.merge(gt_df, pred_df, on='Word_Clean', how='inner')
        
        # Check if we have duplicates (words appearing in multiple clusters? shouldn't happen for hard clustering)
        # If words appear multiple times in ground truth, duplicates might occur.
        # ARI requires 1-to-1 mapping of samples.
        # We'll drop duplicates on 'Word_Clean' to ensure unique samples for evaluation.
        merged_df = merged_df.drop_duplicates(subset=['Word_Clean'])

        n_samples = len(merged_df)
        
        if n_samples < 2:
            print(f"[{method}] Not enough overlapping samples ({n_samples}) to compute ARI.")
            ari = 0.0
        else:
            ari = adjusted_rand_score(merged_df['REMARK'], merged_df['Predicted_Cluster'])
        
        results.append({'Method': method, 'ARI': ari, 'Samples': n_samples})
        # print(f"[{method}] ARI: {ari:.4f} (Samples: {n_samples})")

    # 3. Output Results
    report_lines = []
    report_lines.append("="*40)
    report_lines.append("FINAL ARI SCORES")
    report_lines.append("="*40)
    report_lines.append(f"{'Method':<20} | {'ARI Score':<10} | {'Samples':<10}")
    report_lines.append("-" * 46)
    for res in results:
        report_lines.append(f"{res['Method']:<20} | {res['ARI']:.4f}       | {res['Samples']:<10}")
    report_lines.append("-" * 46)
    
    report_content = "\n".join(report_lines)
    print(report_content)
    
    with open("ari_report.txt", "w") as f:
        f.write(report_content)

if __name__ == "__main__":
    calculate_ari()

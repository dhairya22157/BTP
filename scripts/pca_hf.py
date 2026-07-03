import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_hf_embeddings(model_name, texts, batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()
    
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out = model(**enc)
            last_hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            embeddings.append(pooled.cpu().numpy())
    
    return np.vstack(embeddings).astype(np.float32)

def get_pca_components(embeddings, threshold=0.90):
    n_samples, n_features = embeddings.shape
    n_comp = min(n_samples, n_features)
    
    pca = PCA(n_components=n_comp)
    pca.fit(embeddings)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    components_needed = np.argmax(cumulative_variance >= threshold) + 1
    if cumulative_variance[-1] < threshold:
        components_needed = n_comp
        
    return components_needed

def main():
    df = pd.read_csv(Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'protocol_merged_text.csv')
    df["merged_text"] = df["merged_text"].astype(str)
    texts = df["merged_text"].tolist()
    
    models = {
        "BioBERT": "dmis-lab/biobert-base-cased-v1.1",
        "PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "RecipeBERT": "alexdseo/RecipeBERT",
        # CookBERT script used bert-base-uncased as a placeholder, let's test it too
        "CookBERT": "bert-base-uncased"
    }
    
    results = {}
    for name, repo in models.items():
        print(f"Generating for {name} ({repo})...")
        try:
            emb = get_hf_embeddings(repo, texts)
            comp = get_pca_components(emb)
            results[name] = comp
            print(f"{name}: {comp}")
        except Exception as e:
            print(f"Failed {name}: {e}")
            results[name] = "Failed"
            
    print("\n--- FINAL RESULTS ---")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()

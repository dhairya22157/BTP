import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA, TruncatedSVD
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def get_pca_components(embeddings, threshold=0.90):
    try:
        # If it's sparse, use TruncatedSVD instead of PCA
        if hasattr(embeddings, "toarray") or hasattr(embeddings, "todense"):
            # SVD doesn't center data, so it's not strictly PCA, but it gives variance explained
            # Alternatively, since TF-IDF matrices can be big, convert to dense if small enough
            embeddings = embeddings.toarray()
            
        n_samples, n_features = embeddings.shape
        n_comp = min(n_samples, n_features)
        
        pca = PCA(n_components=n_comp)
        pca.fit(embeddings)
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        # Find minimum index where variance is >= threshold
        components_needed = np.argmax(cumulative_variance >= threshold) + 1
        
        # If the threshold is never reached strictly (e.g. floats), just check max
        if cumulative_variance[-1] < threshold:
            components_needed = n_comp
            
        return components_needed
    except Exception as e:
        print(f"Error computing PCA: {e}")
        return "Not Available"

def main():
    df = pd.read_csv(Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'protocol_merged_text.csv')
    df['cleaned_text'] = df['merged_text'].astype(str).str.strip().str.lower()
    texts = df['cleaned_text'].tolist()
    
    results = {}
    
    # SBERT
    try:
        print("SBERT...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        emb = model.encode(texts)
        results['SBERT'] = get_pca_components(emb)
    except Exception as e:
        print("SBERT failed:", e)
        results['SBERT'] = "Not Available"
        
    # Doc2Vec
    try:
        print("Doc2Vec...")
        sentences = [t.split() for t in texts]
        documents = [TaggedDocument(words=s, tags=[i]) for i, s in enumerate(sentences)]
        d2v = Doc2Vec(vector_size=100, window=5, min_count=1, workers=1, epochs=200)
        d2v.build_vocab(documents)
        d2v.train(documents, total_examples=d2v.corpus_count, epochs=d2v.epochs)
        emb = np.vstack([d2v.dv[i] for i in range(len(sentences))])
        results['Doc2Vec'] = get_pca_components(emb)
    except Exception as e:
        print("Doc2Vec failed:", e)
        results['Doc2Vec'] = "Not Available"
        
    # Word2Vec
    try:
        print("Word2Vec...")
        sentences = [t.split() for t in texts]
        w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=1, epochs=200)
        W = np.vstack([
            np.mean([w2v.wv[w] for w in s if w in w2v.wv], axis=0) if s else np.zeros(w2v.vector_size)
            for s in sentences
        ])
        results['Word2Vec'] = get_pca_components(W)
    except Exception as e:
        print("Word2Vec failed:", e)
        results['Word2Vec'] = "Not Available"
        
    # TF-IDF
    try:
        print("TF-IDF...")
        tfidf_vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        X_tfidf = tfidf_vec.fit_transform(texts)
        results['TF-IDF'] = get_pca_components(X_tfidf.toarray() if X_tfidf.shape[1] < 50000 else "sparse")
    except Exception as e:
        print("TF-IDF failed:", e)
        results['TF-IDF'] = "Not Available"
        
    import json
    with open('pca_results.json', 'w') as f:
        json.dump(results, f)
    
    print("PCA Results:")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1) Setup & data loading
import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "protocol_manual_classification_final.csv"

df = pd.read_csv(DATA_PATH)
print(df.head())
print(df.columns.tolist())

# Assume the column 'Word' holds the protocol token/term. If your protocols are full phrases in 'Word', we'll treat each row as a short document.
texts = df['Word'].astype(str).fillna('').tolist()
ids = df.index.astype(str).tolist()
print(f"Loaded {len(texts)} documents")


# In[2]:


# 2) Preprocessing helpers
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_punct_tbl = str.maketrans('', '', string.punctuation)


def simple_clean(text: str) -> str:
    text = text.lower()
    text = text.translate(_punct_tbl)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens(text: str):
    toks = simple_clean(text).split()
    return [t for t in toks if t not in ENGLISH_STOP_WORDS]

clean_texts = [" ".join(tokens(t)) for t in texts]
print(clean_texts[:10])


# In[3]:


# 3) Embeddings: TF-IDF, Count n-grams, Jaccard shingles, Word2Vec/Doc2Vec, Sentence-BERT
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

# TF-IDF (unigram + bigram)
tfidf_vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
X_tfidf = tfidf_vec.fit_transform(clean_texts)
S_tfidf = cosine_similarity(X_tfidf)
print('TF-IDF shape:', X_tfidf.shape, 'Similarity shape:', S_tfidf.shape)

# Count n-grams (1-3)
count_vec = CountVectorizer(ngram_range=(1,3), min_df=1)
X_count = count_vec.fit_transform(clean_texts)
S_count = cosine_similarity(X_count)
print('Count shape:', X_count.shape)

# Jaccard over character shingles (3-grams) for very short terms
# Build set of char-shingles for each term

def char_shingles(s: str, k=3):
    s = s.replace(' ', '_')
    if len(s) < k:
        return {s}
    return {s[i:i+k] for i in range(len(s)-k+1)}

sets = [char_shingles(t, 3) for t in clean_texts]
S_jaccard = np.zeros((len(sets), len(sets)), dtype=float)
for i in range(len(sets)):
    a = sets[i]
    for j in range(i, len(sets)):
        b = sets[j]
        inter = len(a & b)
        union = len(a | b) if a or b else 1
        sim = inter / union
        S_jaccard[i, j] = S_jaccard[j, i] = sim
print('Jaccard matrix shape:', S_jaccard.shape)

# Word2Vec on tokens, then average word vectors per term
sentences = [t.split() for t in clean_texts]
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=2, epochs=200)
W = np.vstack([
    np.mean([w2v.wv[w] for w in s if w in w2v.wv], axis=0) if s else np.zeros(w2v.vector_size)
    for s in sentences
])
S_w2v = cosine_similarity(W)
print('Word2Vec embedding shape:', W.shape)

# Doc2Vec (treat each term as a document)
documents = [TaggedDocument(words=s, tags=[i]) for i, s in enumerate(sentences)]
d2v = Doc2Vec(vector_size=100, window=5, min_count=1, workers=2, epochs=200)
d2v.build_vocab(documents)
d2v.train(documents, total_examples=d2v.corpus_count, epochs=d2v.epochs)
D = np.vstack([d2v.dv[i] for i in range(len(sentences))])
S_d2v = cosine_similarity(D)
print('Doc2Vec embedding shape:', D.shape)

# Sentence-BERT (robust semantic embeddings even for single tokens)
# Model: all-MiniLM-L6-v2 (~384 dim)
sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
E = sbert_model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
S_sbert = np.matmul(E, E.T)  # cosine sim since normalized
print('Sentence-BERT embedding shape:', E.shape)


# In[5]:


# 4) Hierarchical clustering + plots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

out_dir = ROOT_DIR / 'data' / 'processed' / 'similarity_matrices'
out_dir.mkdir(parents=True, exist_ok=True)

# Utility to cluster and plot from a similarity matrix

def cluster_and_plot(S, title_prefix: str, method='average', max_clusters=8):
    S = np.asarray(S, dtype=float)
    # Numerical safety: enforce symmetry and clip to [-1, 1]
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 1.0)
    S = np.clip(S, -1.0, 1.0)

    # Convert similarity to distance (1 - sim)
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    # Ensure non-negative due to potential float precision
    D = np.maximum(D, 0.0)

    # Condensed distance for linkage
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=method)

    # Dendrogram
    plt.figure(figsize=(10, 4))
    dendrogram(Z, no_labels=True, color_threshold=None)
    plt.title(f"{title_prefix} - Dendrogram ({method})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{title_prefix}_dendrogram.png", dpi=150)
    plt.close()

    # Heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(S, cmap='viridis', vmin=-1 if np.min(S) < 0 else 0, vmax=1)
    plt.title(f"{title_prefix} - Similarity Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / f"{title_prefix}_heatmap.png", dpi=150)
    plt.close()

    # Cluster labels at a chosen max number of clusters
    labels = fcluster(Z, max_clusters, criterion='maxclust')
    pd.Series(labels, index=ids).to_csv(out_dir / f"{title_prefix}_clusters_k{max_clusters}.csv", header=['cluster'])
    # Save similarity matrix
    pd.DataFrame(S, index=ids, columns=ids).to_csv(out_dir / f"{title_prefix}_similarity.csv")

    return labels, Z

# Run for each embedding similarity
labels_tfidf, Z_tfidf = cluster_and_plot(S_tfidf, 'tfidf')
labels_count, Z_count = cluster_and_plot(S_count, 'count')
labels_jacc, Z_jacc = cluster_and_plot(S_jaccard, 'jaccard')
labels_w2v, Z_w2v = cluster_and_plot(S_w2v, 'word2vec')
labels_d2v, Z_d2v = cluster_and_plot(S_d2v, 'doc2vec')
labels_sbert, Z_sbert = cluster_and_plot(S_sbert, 'sbert')

print('Saved outputs to', out_dir.resolve())


# In[6]:


# 5) Optional: evaluate clusters vs. REMARK label
# Produces contingency tables to see alignment with provided categories

def save_cluster_crosstab(labels, name: str, label_col='REMARK'):
    df_tmp = pd.DataFrame({
        'cluster': labels,
        'label': df[label_col].astype(str).fillna('')
    })
    ct = pd.crosstab(df_tmp['cluster'], df_tmp['label'])
    ct.to_csv(out_dir / f'{name}_cluster_vs_{label_col}.csv')
    print(f'Saved crosstab for {name} ->', out_dir / f'{name}_cluster_vs_{label_col}.csv')

save_cluster_crosstab(labels_sbert, 'sbert')
save_cluster_crosstab(labels_tfidf, 'tfidf')


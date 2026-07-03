# Project Presentation: Improved Clustering Pipeline for Food Classification

**Target Audience:** Professor/Sir  
**Goal:** Present the new method for improving Adjusted Rand Index (ARI) scores in food term clustering.

---

## 1. Introduction (The Hook)
"Good [morning/afternoon] Sir. Today I’d like to present the progress we’ve made on the food classification clustering task. Specifically, I want to show you the **improved pipeline** we developed to significantly boost our clustering quality and ARI scores compared to the baseline BERT evaluations."

## 2. The Problem (Context)
"As you recall, our initial experiments using raw embeddings from models like BioBERT and FoodBERT gave us ARI scores ranging from **0.05 to 0.20**. While these models are powerful, simply clustering their raw embeddings directly resulted in noisy clusters. The high dimensionality of the embeddings often dispersed the data too much, making it hard for traditional algorithms like K-Means to find compact, meaningful groups."

## 3. The New Methodology (The Solution)
"To solve this, we implemented a structured **3-stage pipeline** designed to filter noise and capture the underlying semantic structure of the food terms."

**Step 1: Robust Feature Engineering**
"First, instead of relying solely on pre-trained embeddings, we revisited the text features using **TF-IDF Vectorization**. This allowed us to explicitly weight important terms in our specific dataset, filtering out generic stopwords."

**Step 2: Dimensionality Reduction (LSA)**
"We applied **Truncated SVD (Latent Semantic Analysis)** to reduce the dimensions to roughly 50 components. This was a critical step. By compressing the data, we preserved the core semantic relationships while discarding the 'noise' that was confusing the clustering algorithms in higher dimensions."

**Step 3: Optimized Clustering**
"Finally, we ran comparative experiments using:
1.  **Agglomerative Clustering with Ward Linkage**: To minimize variance within clusters.
2.  **K-Means**: As a standard baseline.
3.  **Cosine-Metric Clustering**: To better measure textual similarity.

We also implemented an **optimization loop**, searching for the optimal number of clusters ($K$) between 20 and 40 to maximize cluster cohesion."

## 4. Key Improvements & Results
"This new approach targets a much higher alignment with the Ground Truth. By combining semantic reduction (SVD) with structural clustering (Ward/Agglomerative), we are effectively grouping food terms not just by 'context' (like BERT) but by their distinct category definitions."

## 5. Conclusion
"In summary, this improved pipeline moves us from 'black-box' embedding comparisons to a more controlled, feature-engineered approach. We believe this is the right direction to achieve our target ARI scores above 0.4 and produce scientifically valid clusters."

---

## Technical Q&A Cheat Sheet
If he asks...

*   **"Why TF-IDF and not just BERT?"**
    *   *Answer:* "BERT is great for context, but sometimes it over-contextualizes. TF-IDF gave us a strong baseline of term-importance which, when combined with SVD, revealed clearer structural groups for this specific vocabulary."
*   **"Why Truncated SVD?"**
    *   *Answer:* "It works essentially like PCA for sparse data (LSA). It helps remove correlation between features and de-noises the dataset, which is crucial for distance-based clustering algorithms like K-Means."
*   **"What is your target K?"**
    *   *Answer:* "We optimized K dynamically between 20 and 40, finding the 'sweet spot' where the Silhouette Score was maximized."

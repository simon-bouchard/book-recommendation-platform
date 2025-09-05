# Glossary

- **ALS:** Alternating Least Squares, collaborative filtering for warm users.  
- **LightGBM:** Gradient boosting reranker for refined ranking. Deprecated, it is not actually used anymore. 
- **FAISS:** Approximate nearest-neighbor search library.  
- **Bayesian popularity:** Adjusted rating average balancing quality and count.  
- **Subject-attention embeddings:** Content embeddings weighted by subject importance.  
- **Hybrid similarity:** Slider-controlled mix of subject and behavioral similarity.  
- **Cold-start:** State before you’ve rated 10 books; recommendations fall back to subjects + popularity.  
- **Provenance:** Clear labeling of the source of a recommendation (URLs in Web mode; ALS/Subject/GBM in Internal mode).  

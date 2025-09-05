# How Recommendations Work

The system combines **behavioral signals (ALS)** with **content-based signals (subjects)**. Each has strengths and weaknesses, and the hybrid option lets you balance between them.

## Subject-Based Recommendations
- Focus: **content similarity** based on book subjects.  
- Strengths:
  - Surfaces books that are closely related in **topic or theme**.  
  - Includes books with **very few ratings**, allowing discovery of “hidden gems.”  
- Weaknesses:
  - Subjects can be **noisy or inconsistent** because of imperfect metadata.  
  - Planned enrichment with LLM-based cleaning will improve subject quality over time.  

## ALS (Alternating Least Squares) Recommendations
- Focus: **behavioral similarity** based on patterns of how users rate books.  
- Strengths:
  - Captures **taste overlap** between readers, even across different genres.  
  - Prioritizes **popular books**, which often means stronger signal and higher accuracy.  
  - Filters out books with fewer than **10 ratings** to reduce noise.  
- Weaknesses:
  - May recommend books that are **not exactly the same genre**, but are popular among similar readers.  
  - Less likely to surface niche or obscure titles.  

## Hybrid Recommendations
- Focus: **balanced mix** of subject and ALS signals.  
- Strengths:
  - Combines **content awareness** (subjects) with **behavioral reliability** (ALS).  
  - Filters out books with fewer than **5 ratings**, keeping results cleaner while still surfacing variety.  
- Use case: good default if you want both **genre relevance** and **well-rated options**.  

## Daily Training
- The system retrains every day on a separate server.  
- Personalized recommendations update automatically after training completes.  

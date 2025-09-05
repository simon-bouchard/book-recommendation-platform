# Book Pages

Each book page contains:
- **Basic info:** title, author, year, number of pages, average rating, rating count, ISBN, description.  
- **Rate or mark as read:** with optional comments.  
- **Comments section:** shows existing user comments.  
- **Similar books:** three modes available, all powered by FAISS:  
  - **Subject-based:** similarity from attention-pooled subject embeddings.  
  - **Behavioral:** collaborative filtering using ALS embeddings.  
  - **Hybrid:** adjustable slider mixing subject and behavioral scores.  

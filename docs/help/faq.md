# FAQ

**Q: Why don’t I see personalized recommendations yet?**  
A: You need to rate at least 10 books. Until then, only subject + Bayesian popularity are used. Personalized ALS-based recommendations depend on your ratings **and** the latest daily training run. If you just reached 10 ratings, your personalized results will appear after the next training cycle completes.  

**Q: I already have 10 ratings — why don’t I see personalized recs?**  
A: ALS models are retrained once per day on a separate server. If you just passed 10 ratings, your data will be included in the next run, and personalized recs will show up after that training finishes. Until then, you’ll continue to see subject + popularity recommendations.  

**Q: How do I find similar books?**  
A: On any book page, use the **Similar Books** section.  
- **Subject-based:** closest matches in terms of book content. May include less popular or niche titles.  
- **ALS:** books liked by readers with similar tastes. Tends to favor more popular books with ≥10 ratings.  
- **Hybrid:** mix of both, filtering out books with fewer than 5 ratings.  

**Q: Why are some book details incomplete?**  
A: Metadata is noisy. Improvements are planned through external sources and possible LLM-assisted enrichment.  

**Q: Why do recommendations improve over time?**  
A: More ratings strengthen your profile, and the daily training run incorporates them into ALS embeddings and rerankers. This makes personalized recs more accurate as both your history and the global dataset grow.  

**Q: How does the chatbot remember conversations?**  
A: A small conversation window is kept in Redis (default 3 turns, 2-day expiry). If memory is unavailable, the bot still works but does not remember prior turns.  

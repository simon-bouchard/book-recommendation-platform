# Chatbot

The site includes a **chatbot demo** that acts as a virtual librarian.

## Access
- You don't have to be **logged in** to use the chatbot, but you get stricter quotas if you don't.  
- This prevents anonymous misuse and ensures fair usage.  
- Conversations are still stored only temporarily (Redis + cookie).  

## Modes
- **Web mode (current demo):**  
  - Uses minimal external tools (DuckDuckGo, Wikipedia).  
  - Handles small talk and simple book-related questions.  
  - Can fetch user profile (selected subjects and past interactions) if logged in to provide better context for recommendations.
  - Can access internal docs to give info about the website, help user onboarding and help with technical problems or specific questions.

- **Internal mode (planned, not yet available):**  
  - Will connect directly to the internal recommendation pipeline (ALS, subject embeddings, LightGBM rerankers, Bayesian cold-start).  
  - Designed to provide book suggestions with clear provenance labels (e.g., *ALS*, *Subject*, *GBM*).  
  - Not enabled in the current version.  

## Conversation Memory
- A short rolling history (default: 3 turns) is stored in Redis, scoped to your browser session via a cookie.  
- Memory expires after ~2 days.  
- If Redis is unavailable or cookies are blocked, the chatbot still works but will not remember earlier messages.  

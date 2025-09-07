# Data & Privacy

- No login is required to browse the site, but login is required for personalised recommendations and full chatbot access.
- Conversation history is stored only in memory (Redis), expires automatically, and is tied to your browser cookie.  
- No database writes are performed by the chatbot.  
- User accounts, ratings, and comments (outside the chatbot) are stored securely.  
- Some metadata may be incomplete or inconsistent. Future enrichment with external sources and LLM-based cleaning is planned to improve coverage.  
- Website connection is secured via SSL (HTTPS) and passwords are stored securely and encrtypted.
- No use of thrid party cookies 
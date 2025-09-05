# Profile Page

The profile displays:
- **User info:** username, user ID, email, age, country.  
- **Stats:** number of ratings, books read, favorite subjects.  
- **Edit Profile:** update your details and favorite subjects.  
- **Past interactions:** view books you’ve rated or marked as read.  
- **Recommendations:**  
  - **Subject-based:** blended score between subject similarity and Bayesian popularity, with a slider to adjust the mix.  
  - **Behavioral (ALS):** collaborative filtering using trained embeddings.  
    - This section is shown but grayed out until you have **rated at least 10 books**.  

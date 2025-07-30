import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load the interactions DataFrame
interactions = pd.read_pickle("models/training/data/interactions.pkl")  # Adjust path if needed

# Show basic stats
print(interactions["rating"].describe())

# Compute standard deviation
std_rating = interactions["rating"].std()
print(f"Standard deviation of ratings: {std_rating:.4f}")

print('With Warm ratings only:')

interactions = interactions[interactions["rating"].notnull()]
counts = interactions["user_id"].value_counts()
interactions = interactions[interactions["user_id"].isin(counts[counts >= 10].index)]

print(interactions["rating"].describe())

# Compute standard deviation
std_rating = interactions["rating"].std()
print(f"Standard deviation of ratings: {std_rating:.4f}")
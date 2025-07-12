import pandas as pd

# === STEP 1: Load and clean books.csv ===
df_books = pd.read_csv("data/books.csv")

# Replace invalid work_id values like 'False' or NaN with None
df_books["work_id"] = df_books["work_id"].replace("False", pd.NA)

# Create a consistent work_id per ISBN
# - For books with valid work_id: keep as-is
# - For books with missing work_id: assign 'unmapped_<isbn>'
df_books["work_id"] = df_books.apply(
    lambda row: row["work_id"]
    if pd.notna(row["work_id"])
    else f"unmapped_{row['isbn']}",
    axis=1
)

# Enforce: one work_id per ISBN
isbn_to_work_id = df_books.drop_duplicates("isbn").set_index("isbn")["work_id"].to_dict()

# Reassign all work_id values consistently per ISBN
df_books["work_id"] = df_books["isbn"].map(isbn_to_work_id)

# Drop duplicate work_id rows
df_books = df_books.drop_duplicates(subset="work_id")

# Save updated books.csv
df_books.to_csv("data/books.csv", index=False)

# === STEP 2: Build old→new work_id map for ratings/interactions ===
# You may have inconsistent original work_id usage (like unmapped_)
# So find all work_id→isbn → new work_id mappings
df_books_clean = pd.read_csv("data/books.csv")
isbn_to_work_id_clean = df_books_clean.set_index("isbn")["work_id"].to_dict()

# Reverse map: build old_work_id → new_work_id based on ISBN
df_original = pd.read_csv("data/books.csv")
old_to_new_work_id = dict()
for _, row in df_original.iterrows():
    isbn = row["isbn"]
    old_work_id = row["work_id"]
    new_work_id = isbn_to_work_id_clean.get(isbn)
    if pd.notna(isbn) and old_work_id != new_work_id:
        old_to_new_work_id[old_work_id] = new_work_id

# === STEP 3: Update ratings.csv ===
df_ratings = pd.read_csv("data/ratings.csv")
df_ratings["work_id"] = df_ratings["work_id"].replace(old_to_new_work_id)
df_ratings.to_csv("data/ratings.csv", index=False)

# === STEP 4: Update interactions.csv ===
try:
    df_inter = pd.read_csv("data/interactions.csv")
    df_inter["work_id"] = df_inter["work_id"].replace(old_to_new_work_id)
    df_inter.to_csv("data/interactions.csv", index=False)
    print("✅ interactions.csv updated")
except FileNotFoundError:
    print("ℹ️ interactions.csv not found — skipping")

print("✅ All files cleaned and aligned by consistent work_id")

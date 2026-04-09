# Book Recommendation System
![CI](https://github.com/simon-bouchard/book_recsys/actions/workflows/ci.yml/badge.svg)

A production-grade recommendation engine that supports both **warm users** (with prior ratings) and **cold users** (no history).
The system provides personalized recommendations and item similarity search with very low latency, making it suitable for real-time applications.

---

## Project Overview

At a high level, the system:

- Serves **warm users** using collaborative filtering and metadata-driven reranking
- Serves **cold users** using subject embeddings and popularity priors
- Offers **book similarity search** based on both user behavior and content
- Runs on a fully automated pipeline with daily retraining and hot-reload of new models

---

## Architecture Details

### Warm Users
- Pipeline:
  1. **ALS (Alternating Least Squares)** retrieves top candidate books based on collaborative behavior.
  2. Candidates are reranked with a **LightGBM model** that blends:
     - Learned subject embeddings
     - Metadata features (book stats, overlap counts, cosine similarities)

This approach leverages the strength of ALS for same-author and series recall, while LightGBM provides refined ranking using content and metadata.

### Cold Users
- Pipeline:
  1. **Attention-pooled subject embeddings** compute similarity between a user’s favorite subjects and books.
  2. A **Bayesian popularity prior** balances exploration and robustness (adjustable via a slider).

This allows handling users with no ratings while ensuring recommendations remain meaningful.

### Item Similarity Options
- **ALS (behavioral similarity)**: strong at recalling books from the same author or series, but limited for niche or sparse books.
- **Subject similarity**: more noisy, but better at surfacing hidden gems and underrepresented books.
- **Hybrid strategy**: combines both, with adjustable weights.

### Subject Embeddings
- Learned with a **dual loss**:
  - Regression loss (RMSE on ratings)
  - Contrastive loss (subject co-occurrence patterns)
- **Attention pooling** is applied to weight the most informative subjects for each book, improving similarity quality.

### Automation & Deployment
- **Data pipeline**: normalized SQL schema with users, books, subjects, and interactions.
- **Training server**: runs daily retraining (ALS, LightGBM, aggregates).
- **Inference server**: automatically reloads new models with zero downtime.
- **FastAPI backend**: exposes endpoints and handles db query, authentication, model inference etc.
- **Web frontend**: lightweight app for browsing, searching, rating, and receiving recommendations in real time.

---

## Research & Experiments

In addition to the deployed system, extensive experiments were carried out to study trade-offs between accuracy, latency, and complexity:

- Residual MLPs over dot-product predictions
- Two-tower and three-tower architectures
- Different clustering and regrgession methods on user embeddings
- Gated-fusion mechanisms
- Alternative attention pooling strategies (scalar, per-dimension, transformer-based self-attention)

These studies informed the final production choices.

---

## Tech Stack

- **Python**: core modeling & backend
- **FastAPI**: REST API backend
- **SQL (MySQL/MariaDB)**: normalized data schema
- **LightGBM**: reranking
- **PyTorch**: subject embeddings + attention pooling
- **Implicit**: ALS collaborative filtering
- **FAISS**: similarity search
- **nginx + uvicorn**: deployment
- **Azure VM**: daily training jobs
- **Automation**: CRON-based retraining and model hot-reload

---

## Data & Processing

The original Book-Crossing dataset is noisy and incomplete, with inconsistent ISBNs, duplicate editions, missing metadata, and no subject information.
To build a usable recommendation system, the data was extensively **cleaned, normalized, and enriched** with metadata from Open Library.
Key processing steps:

### 1. ID Normalization & Book Merging
- Original Book-Crossing ratings identify books by ISBN.
- ISBNs were normalized and mapped to **Open Library work IDs**.
- Different editions of the same book were consolidated under a single `work_id`, reducing duplication and ensuring consistent interaction counts.
- Each book in the system is assigned a stable internal integer ID (`item_idx`) for modeling.

### 2. Subject Enrichment & Reduction
- Raw Book-Crossing provides no subject categories.
- Subjects were pulled from **Open Library metadata** for each work.
- Raw extraction yielded ~130,000 unique subject strings.
- Through cleaning, deduplication, and frequency filtering, this set was reduced to ~1,000 meaningful subjects.
- A subject vocabulary (`subject_idx → subject`) is maintained for indexing in models.

### 3. User Data Cleaning
- Ages: extreme or implausible values were removed or bucketed into **age groups**.
- Locations: parsed into **country** and normalized (e.g., removing malformed entries).
- Favorite subjects: for each user, the **top-k subjects** are derived from rated books and stored separately for use in cold-start embeddings.

### 4. Rating Data Cleaning
- Ratings outside the valid range were discarded.
- Duplicate rows were dropped.
- Users/books with too few interactions were filtered out to stabilize training.

### 5. Subject & Metadata Normalization
- Subjects are stored as indexed lists (`subjects_idxs`) with padding/truncation to fixed length.
- Generic categories like *“Fiction”* and *“General”* are excluded from `main_subject` to avoid trivial signals.
- Authors, years, and page counts are cleaned into canonical forms (e.g., “Unknown Author” placeholder, year bucketing, missing pages imputed).

### 6. Aggregate Features
- **Book-level**: number of ratings, average rating, rating standard deviation.
- **User-level**: number of ratings, average rating, rating standard deviation.
- These aggregates are precomputed during export so they remain consistent across training/inference.

---

Result: a normalized SQL schema with **clean IDs, consistent metadata, and a manageable subject vocabulary (~1,000 categories)** that feeds both collaborative and content-based models.

---

## Usage

### Local Development
```bash
# clone repo
git clone https://github.com/simon-bouchard/Book_Recommendation_UI_with_FastAPI
cd book-recsys

# setup env
conda env create -f env.yml
conda activate bookrec-api

# run server
uvicorn app.main:app --reload

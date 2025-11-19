# ops/meilisearch/update_bayes_pop.py
import sys
import os
from pathlib import Path
from meilisearch import Client
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.shared_utils import ModelStore

load_dotenv()
client = Client("http://localhost:7700", os.getenv("MEILI_MASTER_KEY"))
index = client.index("books")
model_store = ModelStore()

BATCH_SIZE = 5000

def refresh_bayes_pop_after_training():
    """Run this after your training pipeline to update bayes_pop for all books"""
    print("Pre-loading model store...")
    item_to_row = model_store.get_item_idx_to_row()
    bayes_tensor = model_store.get_bayesian_tensor()
    
    print("Building bayes_pop updates...")
    updates = []
    for item_idx, row_idx in tqdm(item_to_row.items(), desc="Preparing updates"):
        if row_idx != -1:  # valid index
            updates.append({
                "item_idx": item_idx,
                "bayes_pop": round(float(bayes_tensor[row_idx]), 6)
            })
    
    print(f"Updating bayes_pop for {len(updates)} books in Meilisearch...")
    for i in tqdm(range(0, len(updates), BATCH_SIZE), desc="Indexing"):
        index.update_documents(updates[i:i + BATCH_SIZE])
    
    print(f"✅ Updated bayes_pop for {len(updates)} books")

if __name__ == "__main__":
    refresh_bayes_pop_after_training()

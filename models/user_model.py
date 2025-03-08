import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Embedding
from fastapi import HTTPException
from fastai.data.core import DataLoaders
from fastai.layers import sigmoid_range
import os
import sys
from dotenv import load_dotenv
import asyncio
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import SessionLocal
from app.table_models import User, Book, Rating

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
model_file_path = os.path.join(script_dir, "user_model_20.pth")
dls_path = os.path.join(script_dir, 'dls.pkl')

model = None

with open(dls_path, 'rb') as f:
    dls = pickle.load(f)

def reload_user_model():
    print('Model reload...')

    global model

    class CollabNN(Module):
        def __init__(self, user_sz, item_sz, y_range=(1,10.5), n_act=100, dropout=0.2):
            super().__init__()
            self.user_factors = Embedding(*user_sz)
            self.item_factors = Embedding(*item_sz)
            self.layers = nn.Sequential(
                nn.Linear(user_sz[1]+item_sz[1], n_act),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(n_act, n_act // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(n_act // 2, n_act // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(n_act // 4, n_act // 8),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(n_act // 8, n_act // 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(n_act // 16, 1))
            self.y_range = y_range
    
        def forward(self, x):
            embs = self.user_factors(x[:,0]),self.item_factors(x[:,1])
            x = self.layers(torch.cat(embs, dim=1))
            return sigmoid_range(x, *self.y_range)

    model = CollabNN((7512, 300), (674, 300), n_act=300)
    model.load_state_dict(torch.load(model_file_path))
    model.eval()

    print('Model reloaded')

async def get_user_recommendations(user, _id: bool = True, top_n: int = 5):

    if model is None:
        reload_user_model()
 
    db = SessionLocal()

    if _id:
        user_id = int(user)
    else:
        user_entry = db.query(User).filter(User.username == user).first()
        if not user_entry:
            return {'error': 'User not found'}
        user_id = user_entry.id
   
    books_query = db.query(Book.isbn, Book.title, Book.author).all()
    books = pd.DataFrame(books_query, columns=['isbn', 'title', 'author'])

    all_book_titles = list(dls.classes['title'].o2i.keys())

    book_indices = torch.tensor(list(dls.classes['title'].o2i.values()), dtype=torch.int64)

    try:
        user_tensor = torch.full((len(book_indices),), int(user_id), dtype=torch.int64)
        batch = torch.stack([user_tensor, book_indices], dim=1)

        model.eval()
        with torch.no_grad():
            predictions = model(batch)

        predictions = predictions.squeeze().numpy()

    except IndexError:
        raise ValueError("User index out of range. The user_id might not exist in the model.")
    except ValueError as ve:
        raise ve  
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during recommendation: {str(e)}")


    recommendations = pd.DataFrame({
        'title': all_book_titles,
        'predicted_rating': predictions
    })

    recommendations = recommendations.merge(books, on='title', how='left')

    rated_books = db.query(Rating.book_isbn).filter(Rating.user_id == user_id).all()
    rated_books = {isbn for (isbn,) in rated_books}
           
    recommendations = recommendations[~recommendations['title'].isin(rated_books)]

    recommendations = recommendations.sort_values(by='predicted_rating', ascending=False)

    print(recommendations)
    return recommendations[['title', 'isbn', 'author', 'predicted_rating']].head(top_n).to_dict(orient='records')

async def test():
    result = await get_user_recommendations(314)
    print(result)


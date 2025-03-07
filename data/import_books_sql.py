import pandas as pd
from pymongo import MongoClient
import os
import sys
from dotenv import load_dotenv
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#from app.models import Books

load_dotenv()

client = MongoClient(os.getenv('MONGO_URI'))

db = client['book-recommendation']

books = db['Books']

file_path = os.path.join(os.getcwd(), 'BX-Books.csv')

df = pd.read_csv('BX-Books.csv', encoding='ISO-8859-1', sep=';', quotechar='"', engine='python', on_bad_lines='skip')

df.rename(columns={
    'Book-Title': 'title',
    'ISBN': 'isbn',
    'Book-Author': 'author',
    'Year-Of-Publication': 'year',
    'Publisher': 'publisher'
}, inplace=True)

data = df.to_dict(orient='records')

#validated_ratings = [Rating(**rating).dict(by_alias=True) for rating in data]

books.insert_many(data)

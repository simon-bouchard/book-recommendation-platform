import pandas as pd
import numpy as np
import os
import sys
from dotenv import load_dotenv
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fuzzywuzzy import process
import pycountry
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import database_url
from app.table_models import user

valid_countries = [country.name for country in pycountry.countries]

df = pd.read_csv('./BX-Users.csv', encoding='ISO-8859-1', sep=';')
df = df.replace({np.nan: None})
#df['country'] = df['Location'].str.split(',').str[-1].str.strip()
def extract_country(location):
    if pd.isna(location):  
        return None
    parts = location.split(',')  
    return parts[-1].strip()  

def correct_country(location):
    if pd.isna(location):  
        return None

    part = extract_country(location)
    if not part.strip(' /-?()*&.').strip('-*'):
        return None
    match, score = process.extractOne(part, valid_countries)
    if score >= 80:  
        return match
    return None 


print(df.head())


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

for _, row in df.iterrows():
    user = User(
        id=row['User-ID'],
        location=row['Location'],
        age=row['Age'],
    )
    db.merge(user)

db.commit()
db.close()

print('Books imported successfully')

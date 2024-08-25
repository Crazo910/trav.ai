import pandas as pd
import json
import os
import sqlite3


df = pd.read_csv('madrid_landmarks.csv', dtype={
    'Landmark_ID': 'int64',
    'Name': 'object',
    'Type': 'object',
    'Address': 'object',
    'Latitude': 'float64',
    'Longitude': 'float64'
})

connection = sqlite3.connect("landmarks1.db")
df.to_sql(name="landmarks", con=connection)
import os
import json
import pickle
import time

from pathlib import Path
from dotenv import load_dotenv
from google.generativeai import configure, embed_content
from tqdm import tqdm

import google.generativeai as genai

# File paths
BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR.parent / 'data'
input_json = data_dir / 'meme_texts.json'
output_pkl = data_dir / 'meme_embeddings.pkl'

# Load environment
load_dotenv(BASE_DIR.parent / '.env')
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")

configure(api_key=api_key)

# Load meme texts from JSON file
with open(input_json, 'r', encoding='utf-8') as f:
    meme_data = json.load(f)

# Embed and collect meme texts
emebedded_memes = []

print(f"Embedding {len(meme_data)} meme texts...")

for idx, (file_name, text) in enumerate(tqdm(meme_data.items(), desc='Processing Memes')):
    text = text.strip()

    if not text:
        continue
    try:
        response = embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        embedding = response["embedding"]
        emebedded_memes.append({
            "filename": file_name,
            "text": text,
            "embedding": embedding
        })
        if idx % 100 == 0:
            print(f"Embedded {idx} / {len(meme_data)}")

        time.sleep(0.1) # To avoid hitting rate limits

    except Exception as e:
        print(f"Error embedding {file_name}: {e}")

# Save to pickle file
with open(output_pkl, 'wb') as f:
    pickle.dump(emebedded_memes, f)

print(f"Saved embeddings to {output_pkl}")
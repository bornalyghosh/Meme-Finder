import os
import pickle
import json
import time
import numpy as np

from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

import google.generativeai as genai
from google.generativeai import embed_content

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")
genai.configure(api_key=GOOGLE_API_KEY)

# Paths
BASE_DIR = Path(__file__).resolve().parent
EMBEDDING_FILE = BASE_DIR.parent / 'data' / 'meme_embeddings.pkl'
IMAGE_DIR = BASE_DIR.parent / 'images'

# Load embedded memes
with open(EMBEDDING_FILE, 'rb') as f:
    embedded_memes = pickle.load(f)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embed_query(text):
    response = embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return response["embedding"]

def search_memes(query, top_k=5):
    query_embed = embed_query(query)
    scored_memes = []

    for meme in embedded_memes:
        sim = cosine_similarity(query_embed, meme["embedding"])
        scored_memes.append((sim, meme))

    top_results = sorted(scored_memes, key=lambda x: x[0], reverse=True)[:top_k]
    return top_results

if __name__ == "__main__":
    print("Meme Finder - Semantic Search Engine")
    query = input("Enter your search: ").strip()

    print("\nTop Matches\n")
    for score, meme in search_memes(query):
        print(f"{meme['filename']} | Score: {score: 0.4f}")
        print(f"{meme['text'][:300].strip()}...\n")
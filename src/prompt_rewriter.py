import os
import json
import pickle
import hashlib

from dotenv import load_dotenv
from pathlib import Path

import google.generativeai as genai


# Paths
BASE_DIR = Path(__file__).resolve().parent
CACHE_FILE = BASE_DIR.parent / 'data' / 'prompt_cache.json'

# Load environment variables
load_dotenv(BASE_DIR.parent / '.env')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")
client = genai.configure(api_key=GOOGLE_API_KEY)


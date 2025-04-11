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
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# for model in genai.list_models():
#     print(f"Model: {model.name}, Description: {model.description}")


# Load cache
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        prompt_cache = json.load(f)
else:
    prompt_cache = {}

# Prompt hashing
def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()

# Rewrite prompt function
def rewrite_promp_with_gemini(user_input: str)  -> str:
    prompt_hash = hash_prompt(user_input)

    # Return cached if available
    if prompt_hash in prompt_cache:
        return prompt_cache[prompt_hash]
    
    prompt = f"""You are helping improve a meme search engine.
        Given the user input: \"{user_input}\", rewrite it as a descriptive, friendly, and humor-aware search query that describes what kind of meme the user is looking for.

        The rewritten query should be:
        - Clear
        - Context-rich
        - Related to meme language or situations

        Output just the rewritten query."""
    
    try:
        response = model.generate_content(prompt)
        rewriten = response.text.strip()
        prompt_cache[prompt_hash] = rewriten

        # Save updated cache
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(prompt_cache, f, indent=4, ensure_ascii=False)
        return rewriten
    except Exception as e:
        print(f"[Gemini Error] using raw input due to: {e}")
        return user_input.strip()
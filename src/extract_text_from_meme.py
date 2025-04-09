import os
import json
import pytesseract

from PIL import Image, ImageFile
from pathlib import Path
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set the path to the Tesseract-OCR executable
# Note: Adjust this path according to your Tesseract installation location
# For example, on Windows it might be something like:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define paths
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR.parent / 'images'
OUTPUT_FILE = BASE_DIR.parent / 'data' / 'meme_texts.json'

# Ensure output directory exists
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Load existing data if available (to avoid overwriting)
if OUTPUT_FILE.exists():
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        meme_texts = json.load(f)
else:
    meme_texts = {}

# Process all images
image_files = list(IMAGE_DIR.glob('*.jpg')) + list(IMAGE_DIR.glob('*png')) + list(IMAGE_DIR.glob('*jpeg')) + list(IMAGE_DIR.glob('*webp'))
print(f"Found {len(image_files)} image files in 'images/'")

for img_path in tqdm(image_files, desc="Extracting text from memes"):
    img_name = img_path.name

    # Skip if already processed
    if img_name in meme_texts:
        continue
    try:
        img = Image.open(img_path)
        text = pytesseract.image_to_string(img)
        meme_texts[img_name] = text.strip()
    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        meme_texts[img_name] = ""

# Save to JSON file
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(meme_texts, f, indent=4, ensure_ascii=False)

print(f"Text extraction complete. Saved to {OUTPUT_FILE.relative_to(BASE_DIR.parent)}")
from pathlib import Path
import json
from typing import List
from lib.types import Movie
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
PROMPT_PATH = PROJECT_ROOT / "cli" / "lib" / "prompts"

def load_movies() -> List[Movie]:
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    return data['movies']

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
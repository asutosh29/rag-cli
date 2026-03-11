from pydantic import BaseModel
from typing import TypedDict, List
class Movie(TypedDict):
    id: int
    title: str
    description: str

class MovieDataSet(TypedDict):
    movies: List[Movie]

class SimilarityResult(TypedDict):
    score: float
    title: str
    description: str
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

class ChunkMetaData(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int

class AllChunkMetaData(TypedDict):
    chunks: List[ChunkMetaData]
    total_chunks: int

class ChunkSimilarityResult(TypedDict):
    chunk_idx: int
    movie_idx: int
    score: float

class SearchChunkResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: dict
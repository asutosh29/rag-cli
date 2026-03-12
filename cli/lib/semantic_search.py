from sentence_transformers import SentenceTransformer
from typing_extensions import List
from lib.types import Movie, MovieDataSet, SimilarityResult, ChunkMetaData, AllChunkMetaData, ChunkSimilarityResult, SearchChunkResult
import numpy as np
import os
from pathlib import Path
import json
from lib.utils import load_movies , cosine_similarity
import re
from typing import DefaultDict
class SemanticSearch:
    def __init__(self, model_string = "all-MiniLM-L6-v2") -> None:
        model = SentenceTransformer(model_string)
        self.model = model
        self.embeddings = None
        self.documents = None
        self.embedding_path = Path("cache/movie_embeddings.npy")
        self.document_map = dict()

    def generate_embedding(self, text: str) -> list:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty for embedding generation")
        
        embeddings = self.model.encode([text.strip()], show_progress_bar=True)
        return embeddings[0]
    
    def build_embeddings(self, documents: List[Movie]):
        self.documents = documents
        self.document_map = {}
        movie_strings = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save(file=self.embedding_path,arr=self.embeddings)

        return self.embeddings
    
    def load_or_create_embeddings(self, documents: List[Movie]):
        '''
        Create documents if they dont exist. 
        Otherwise load them from the cache
        '''
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        if self.embedding_path.exists():
            self.embeddings = np.load(self.embedding_path)
            if len(self.documents) == len(self.embeddings):
                return self.embeddings
        return self.build_embeddings(documents)
    
    def search(self, query, limit) -> List[SimilarityResult]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        cosine_similarities = []
        embeddings = self.load_or_create_embeddings(self.documents)
        for doc_embedding, doc in zip(embeddings, self.documents):
            cosine_similarities.append((cosine_similarity(doc_embedding, query_embedding),doc))
        
        cosine_similarities.sort(key=lambda x: x[0], reverse=True)
        results = []
        for i in range(limit):
            record =  cosine_similarities[i]
            sc = record[0]
            doc: Movie = record[1]
            results.append({
                "score": sc,
                "title":doc["title"],
                "description": doc["description"] 
            })
        return results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata: AllChunkMetaData | None = None
        self.chunk_embedding_path = Path("cache/chunk_embeddings.npy")
        self.chunk_metadata_path = Path("cache/chunk_metadata.json")

    def build_chunk_embeddings(self, documents: List[Movie]):
        self.documents = documents
        self.document_map = {}
        chunks: List[str] = []
        chunk_metadata: List[ChunkMetaData]=[] 
        for doc in documents:
            if doc["description"].strip() == "":
                continue
            self.document_map[doc["id"]] = doc
            doc_chunks = semantic_chunk(doc["description"],1,4)
            for i, doc_chunk in enumerate(doc_chunks):
                meta: ChunkMetaData = {
                    "movie_idx": doc["id"],
                    "chunk_idx": i,
                    "total_chunks": len(doc_chunk)
                }
                chunks.append(doc_chunk)
                chunk_metadata.append(meta)
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True) 
        self.chunk_metadata = chunk_metadata  
        np.save(file=self.chunk_embedding_path, arr=self.chunk_embeddings)
        with open(self.chunk_metadata_path,'w') as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks":len(self.chunk_embeddings)},f, indent=2)
        
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: List[Movie]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            if doc["description"].strip() == "":
                continue
            self.document_map[doc["id"]] = doc
        if self.chunk_embedding_path.exists() and self.chunk_metadata_path.exists():
            print("Loading from cache...")
            # Load Embeddings
            self.chunk_embeddings = np.load(self.chunk_embedding_path)
            # Load Metadata
            with open(self.chunk_metadata_path, 'r') as f:
                self.chunk_metadata = json.load(f)
            
            print(f"Loaded {len(self.chunk_embeddings)} chunks embeddings")
            return self.chunk_embeddings
    
        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit=5):
        query_embedding = self.generate_embedding(query)
        chunk_scores: List[ChunkSimilarityResult] = []
        movie_idx_to_scores = DefaultDict(lambda: 0)
        # Compute all similarity scores
        for idx in range(len(self.chunk_embeddings)):
            chunk_embedding = self.chunk_embeddings[idx]
            metadata = self.chunk_metadata["chunks"][idx]
            midx, cidx = metadata["movie_idx"], metadata["chunk_idx"]

            sim_score = cosine_similarity(query_embedding, chunk_embedding)
            chunk_similarity_result: ChunkSimilarityResult = {
                "chunk_idx": cidx,
                "movie_idx": midx,
                "score": sim_score
            }
            # Store score metadata
            chunk_scores.append(chunk_similarity_result)
                        
            # Calc the aggregate scores.
            # This is required since in the end we want to pull out a MOVIE based the highest sim score
            # So natual choice for aggregation is max pooling
            movie_idx_to_scores[midx] = max(sim_score, movie_idx_to_scores[midx]) # defaultdict assigns 0 to any new key

        movie_scores_sorted = sorted(movie_idx_to_scores.items(), key=lambda x: x[1], reverse=True)
        res: List[SearchChunkResult] = []
        for midx, score in movie_scores_sorted[:limit]:
            doc: Movie = self.document_map[midx]
            search_chunk_results: SearchChunkResult = {
            "id": doc["id"],
            "title": doc["title"],
            "document": doc["description"],
            "score": round(score, 4),
            "metadata": {}}
            res.append(search_chunk_results)

        return res

## Util functions
def fixed_size_chunking(text: str, chunk_size=200) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0,len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

def overlap_chunking(text: str, overlap=0,chunk_size=200) -> List[str]:
    words = text.split()
    chunks = []
    if chunk_size < overlap:
        raise ValueError("overlap can't be more than chunk_size")
    
    step_size = chunk_size - overlap
    for i in range(0,len(words), step_size):
        chunk_words = words[i:i+chunk_size]
        if len(chunk_words) <= overlap:
            continue
        chunks.append(" ".join(chunk_words))
    
    return chunks

def semantic_chunk(text: str, overlap=0,max_chunk_size=4)-> List[str]:
    senteces = re.split(r"(?<=[.!?])\s+",text)
    chunks = []
    if max_chunk_size < overlap:
        raise ValueError("overlap can't be more than max_chunk_size")
    
    step_size = max_chunk_size - overlap
    for i in range(0,len(senteces), step_size):
        chunk_words = senteces[i:i+max_chunk_size]
        if len(chunk_words) <= overlap:
            continue
        chunks.append(" ".join(chunk_words))
    
    return chunks


## API Functions
def search_chunk_documents(query, limit=5):
    ss = ChunkedSemanticSearch()
    movies = load_movies()
    ss.load_or_create_chunk_embeddings(movies)
    results = ss.search_chunks(query, limit)
    for i, res in enumerate(results):
        print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
        print(f"   {res['document'][:100]}...")

def embed_chunks():
    movies = load_movies()
    css = ChunkedSemanticSearch()
    embeddings =css.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")

def search_documents(query, limit=5):
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)

    results = ss.search(query, limit)
    for i, res in enumerate(results):
        print(f"{i}. {res['title']} ({res['score']})\n{res['description'][:100]}...")

def chunk_text(text, overlap=0, max_chunk_size=4):
    chunks = semantic_chunk(text, overlap,max_chunk_size)
    print(f"Sematic {max_chunk_size} sentences")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")

def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    ss = SemanticSearch()
    documents: MovieDataSet = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
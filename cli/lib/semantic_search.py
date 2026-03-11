from sentence_transformers import SentenceTransformer
from typing_extensions import List
from lib.types import Movie, MovieDataSet, SimilarityResult
import numpy as np
import os
from pathlib import Path
import json
from lib.utils import load_movies , cosine_similarity
import re

class SemanticSearch:
    def __init__(self, model_string = "all-MiniLM-L6-v2"):
        model = SentenceTransformer(model_string)
        self.model = model
        self.embeddings = None
        self.documents = None
        self.embedding_path = Path("cache/movie_embeddings.npy")
        self.document_map = dict()

    def generate_embedding(self, text: str) -> list:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty for embedding generation")
        
        embeddings = self.model.encode([text.strip()])
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

def semantic_chunk(text: str, overlap=0,max_chunk_size=4):
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
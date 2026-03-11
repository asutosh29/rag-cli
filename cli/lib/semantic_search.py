from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, model_string = "all-MiniLM-L6-v2"):
        model = SentenceTransformer(model_string)
        self.model = model

def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")
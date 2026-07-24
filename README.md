# Road to RAG

A **semantic movie search CLI** built for learning Retrieval-Augmented Generation (RAG) concepts. Search a movie dataset using natural language — powered by local sentence-transformers embeddings and LLM query enhancement via Groq.

## Features

- **Semantic search** — Find movies by meaning, not keywords, using `all-MiniLM-L6-v2` embeddings
- **Chunked search** — Sentence-level chunking with overlap for more precise matching on long descriptions
- **LLM-enhanced queries** — Fix typos, rewrite vague queries into search terms, or expand with synonyms
- **Smart caching** — Embeddings cached to disk; auto-detects dataset changes

## Installation

**Prerequisites:** Python ≥ 3.12, [uv](https://docs.astral.sh/uv/), and a Groq API key.

```bash
uv sync
```

Create a `.env` file with your Groq API key:

```
GROQ_API_KEY=gsk_your_key_here
```

## Usage

All commands run from the project root:

```bash
python cli/cli.py <command> [args]
```

### Search

```bash
# Whole-document semantic search
python cli/cli.py search "a scary movie with a bear" --limit 5

# Chunked search (sentence-level matching)
python cli/cli.py search_chunk "time travel adventure"

# With query enhancement
python cli/cli.py search_chunk "horrer film" --limit 5 --enhance spell
python cli/cli.py search_chunk "that bear movie where leo gets attacked" --limit 3 --enhance rewrite
python cli/cli.py search_chunk "scifi time travel" --limit 5 --enhance expand
```

### Utilities

```bash
python cli/cli.py verify              # Test model loading
python cli/cli.py embed_text "text"   # Generate an embedding vector
python cli/cli.py embedquery "query"  # Embed a query string
python cli/cli.py verify_embeddings   # Check cached embeddings
python cli/cli.py embed_chunks        # Build chunk embeddings
python cli/cli.py chunk "text"        # Demo sentence chunking
```

### Commands

| Command | Description |
|---|---|
| `search "query" --limit N` | Semantic search over full movie descriptions |
| `search_chunk "query" --limit N --enhance spell\|rewrite\|expand` | Chunked search with optional LLM query enhancement |
| `verify` | Verify the embedding model loads correctly |
| `embed_text "text"` | Print embedding vector for given text |
| `verify_embeddings` | Check that cached embeddings match the dataset |
| `embedquery "query"` | Embed a query string and show its vector |
| `chunk "text" --overlap N --max-chunk-size N` | Demonstrate sentence-level chunking |
| `embed_chunks` | Build and cache chunk embeddings |
| `test "query"` | Run test operations |

## Project Structure

```
cli/
├── cli.py                  # CLI entry point (argparse)
└── lib/
    ├── semantic_search.py  # SemanticSearch & ChunkedSemanticSearch classes
    ├── llm.py              # Groq LLM integration for query enhancement
    ├── types.py            # TypedDict types
    ├── utils.py            # Data loading, prompt loading, cosine similarity
    └── prompts/
        ├── spell_check.md  # Prompt: fix typos in queries
        ├── rewrite_query.md # Prompt: convert vague queries to search terms
        └── expand_query.md # Prompt: add synonyms and related terms
data/
└── movies.json             # ~50 movies with title and description
cache/
├── movie_embeddings.npy    # Cached full-document embeddings
├── chunk_embeddings.npy    # Cached chunk embeddings
└── chunk_metadata.json     # Chunk-to-movie mapping metadata
```

## How It Works

1. **Embed** — Movie descriptions and titles are converted to vectors using `sentence-transformers/all-MiniLM-L6-v2`
2. **Cache** — Embeddings are saved to `cache/` and reused until the dataset changes
3. **Search** — User query is embedded and compared via cosine similarity against all movie vectors
4. **Chunk** — For chunked search, descriptions are split into sentences, each embedded independently, then results are **max-pooled** (the best-matching chunk determines the movie's score)
5. **Enhance** — Optionally, the query is transformed by the LLM before embedding: spell check, rewrite, or expansion

## Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | Local text embeddings (MiniLM) |
| `langchain-groq` | Groq API integration for query enhancement |
| `numpy` | Vector math and cosine similarity |
| `python-dotenv` | Load `.env` for API keys |
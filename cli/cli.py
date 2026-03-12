#!/usr/bin/env python3
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search_documents, chunk_text, overlap_chunking, embed_chunks, search_chunk_documents
from lib.llm import spell_check, test
import argparse

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    subparser.add_parser("verify",help="Verify the loading of embedding model")
    embed_subparser = subparser.add_parser("embed_text", help="Generate text embedding for given text")
    embed_subparser.add_argument("text", type=str, help="Text to be encoded")

    subparser.add_parser("verify_embeddings",help="verify if the embeddings works properly")

    query_parser = subparser.add_parser("embedquery", help="embed the query and returns the embedding")
    query_parser.add_argument("query", type=str, help="input query string")
    
    search_parser = subparser.add_parser("search", help="Search nearest documents")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit",type=int, default=5, help="Number of top results")

    search_chunk_parser = subparser.add_parser("search_chunk", help="Search nearest documents using chunks")
    search_chunk_parser.add_argument("query", type=str, help="Search query")
    search_chunk_parser.add_argument("--limit",type=int, default=5, help="Number of top results")
    search_chunk_parser.add_argument("--enhance",type=str,choices=["spell", "rewrite","expand"], help="Corrects spelling mistakes")


    chunk_parser = subparser.add_parser("chunk", help="Chunk the documents according to given chunking parameter")
    chunk_parser.add_argument("text", type=str,help="text to be chunked")
    chunk_parser.add_argument("--overlap", type=int,default=0,help="number of words to be overlapped between chunks")
    chunk_parser.add_argument("--max-chunk-size", type=int, default=4,help="number of words to be chunked in a single chunk")

    subparser.add_parser("embed_chunks", help="Embed documents with chunking")

    test_parser = subparser.add_parser("test",help="Run test commands")
    test_parser.add_argument("query",type=str,help="Text to be spell checked")

    args = parser.parse_args()

    match args.command:
        case "test":
            test(args.query)
        case "search_chunk":
            search_chunk_documents(args.query, args.limit, args.enhance)
        case "embed_chunks":
            embed_chunks()
        case "chunk":
            chunk_text(args.text, args.overlap, args.max_chunk_size, )
        case "search":
            search_documents(args.query, args.limit)
        case "verify":
            verify_model()
        case "embedquery":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case _:
            parser.print_help()
        

if __name__ == "__main__":
    main()

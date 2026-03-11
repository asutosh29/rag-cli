#!/usr/bin/env python3
from lib.semantic_search import verify_model
import argparse

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    subparser.add_parser("verify",help="Verify the loading of embedding model")
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case _:
            parser.print_help()
        

if __name__ == "__main__":
    main()

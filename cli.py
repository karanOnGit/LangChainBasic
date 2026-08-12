#!/usr/bin/env python3
"""
Interactive Command Line Interface (CLI) for Conversational RAG with Groq & LangChain.
Supports document loading, vector store selection, and multi-turn streaming conversations.
"""

import sys
import os
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import RAGConfig, AVAILABLE_GROQ_MODELS, SUPPORTED_VECTOR_STORES
from src.loaders import DocumentLoaderManager
from src.splitters import TextSplitterManager
from src.embeddings import get_embedding_model
from src.vectorstores import VectorStoreManager
from src.memory import MemoryManager
from src.rag_chain import ConversationalRAGChain

console = Console()


def display_welcome_banner():
    console.print(
        Panel.fit(
            "[bold cyan]⚡ Karan Bhardwaj — Conversational AI & Resume RAG[/bold cyan]\n"
            "[bold yellow]Full Stack & AI Systems Engineer[/bold yellow] | [dim]karanbhardwaj.in[/dim]\n"
            "[dim]Powered by LangChain LCEL & Groq (LLaMA 3.3 70B)[/dim]",
            border_style="cyan",
        )
    )


def display_sources(documents):
    if not documents:
        return
    table = Table(title="🔍 Retrieved Source Contexts", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Source", style="cyan", width=25)
    table.add_column("Snippet Preview", style="white")

    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source") or doc.metadata.get("file_name", "Unknown")
        snippet = doc.page_content.replace("\n", " ").strip()
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        table.add_row(str(i), source, snippet)

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Conversational RAG CLI powered by LangChain and Groq")
    parser.add_argument("-f", "--file", type=str, help="Path to document file (PDF, TXT, DOCX, MD)")
    parser.add_argument("-u", "--url", type=str, help="Web URL to load and index")
    parser.add_argument(
        "-s",
        "--store",
        type=str,
        default="faiss",
        choices=SUPPORTED_VECTOR_STORES,
        help="Vector store backend (faiss, chroma, pinecone)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="llama-3.3-70b-versatile",
        help="Groq LLM model name",
    )
    parser.add_argument("-k", "--top-k", type=int, default=4, help="Number of retrieved chunks")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Document chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Document chunk overlap")

    args = parser.parse_args()
    display_welcome_banner()

    # Load configuration
    config = RAGConfig()
    api_key = config.groq_api_key

    if not api_key:
        api_key = Prompt.ask("[bold yellow]Enter your Groq API Key[/bold yellow]", password=True)
        if not api_key:
            console.print("[bold red]Error: Groq API key is required.[/bold red]")
            sys.exit(1)

    # Document Ingestion
    documents = []
    if args.file:
        console.print(f"[bold green]Loading document:[/bold green] {args.file}")
        documents = DocumentLoaderManager.load_from_file(args.file)
    elif args.url:
        console.print(f"[bold green]Scraping URL:[/bold green] {args.url}")
        documents = DocumentLoaderManager.load_from_url(args.url)
    else:
        # Default to Karan Bhardwaj's official resume
        sample_path = Path(__file__).parent / "sample_data" / "karan_bhardwaj_resume.md"
        if sample_path.exists():
            console.print(f"[bold yellow]No file specified. Loading default:[/bold yellow] [bold green]{sample_path.name}[/bold green] (Karan Bhardwaj's Resume)")
            documents = DocumentLoaderManager.load_from_file(str(sample_path))
        else:
            console.print("[bold red]Please provide a document file (-f) or URL (-u).[/bold red]")
            sys.exit(1)

    # Chunking
    console.print("[dim]Splitting document into chunks...[/dim]")
    splitter = TextSplitterManager(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = splitter.split_documents(documents)
    console.print(f"[bold green]✓[/bold green] Created [bold cyan]{len(chunks)}[/bold cyan] chunks.")

    # Embeddings
    console.print("[dim]Initializing embedding model...[/dim]")
    embeddings = get_embedding_model(config.embedding_model_name)

    # Vector Store
    console.print(f"[dim]Building [bold cyan]{args.store.upper()}[/bold cyan] vector store...[/dim]")
    persist_dir = config.chroma_persist_dir if args.store == "chroma" else config.faiss_persist_dir
    vector_store = VectorStoreManager.create_vector_store(
        store_type=args.store,
        documents=chunks,
        embeddings=embeddings,
        persist_dir=persist_dir,
        pinecone_api_key=config.pinecone_api_key,
        pinecone_index_name=config.pinecone_index_name,
    )
    retriever = VectorStoreManager.get_retriever(vector_store, k=args.top_k)

    # Memory & RAG Pipeline
    memory = MemoryManager(max_history_messages=20)
    rag_chain = ConversationalRAGChain(
        retriever=retriever,
        groq_api_key=api_key,
        model_name=args.model,
        temperature=config.temperature,
        memory_manager=memory,
    )

    console.print(f"\n[bold green]✓ RAG System ready![/bold green] (Model: [cyan]{args.model}[/cyan] | VectorStore: [cyan]{args.store.upper()}[/cyan])")
    console.print("[dim]Type [bold white]/clear[/bold white] to reset chat memory, or [bold white]/exit[/bold white] to quit.\n[/dim]")

    session_id = "cli_session"

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
            if not user_input:
                continue

            if user_input.lower() in ["/exit", "exit", "quit", "q"]:
                console.print("[dim]Goodbye![/dim]")
                break
            elif user_input.lower() == "/clear":
                rag_chain.clear_history(session_id)
                console.print("[yellow]Conversation memory cleared.[/yellow]")
                continue
            elif user_input.lower() == "/stats":
                history = rag_chain.get_history(session_id)
                console.print(f"[dim]Total messages in session memory: {len(history)}[/dim]")
                continue

            console.print("\n[bold green]Assistant[/bold green]: ", end="")
            
            sources = []
            for chunk in rag_chain.stream_query(user_input, session_id=session_id):
                if chunk["type"] == "token":
                    console.print(chunk["content"], end="")
                elif chunk["type"] == "sources":
                    sources = chunk["documents"]

            console.print("\n")
            if sources:
                display_sources(sources)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session terminated.[/dim]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    main()

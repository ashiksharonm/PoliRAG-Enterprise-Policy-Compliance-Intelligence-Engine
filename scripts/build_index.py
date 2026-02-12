#!/usr/bin/env python3
"""Build or rebuild vector indexes."""

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking.chunker import TablePreservingChunker
from src.embeddings.service import EmbeddingService
from src.ingestion.manager import DocumentManager
from src.observability.logging import setup_logging
from src.vectorstore.bm25_store import BM25Store
from src.vectorstore.faiss_store import FAISSVectorStore

console = Console()


@click.command()
@click.option(
    "--index-name",
    "-i",
    default="main",
    help="Index name"
)
@click.option(
    "--tenant-id",
    "-t",
    default=None,
    help="Filter by tenant ID (default: all tenants)"
)
@click.option(
    "--rebuild/--no-rebuild",
    default=False,
    help="Rebuild index from scratch"
)
def main(index_name: str, tenant_id: str, rebuild: bool):
    """Build or rebuild vector and BM25 indexes.
    
    Examples:
    
        # Build index for all documents
        python scripts/build_index.py
        
        # Build index for specific tenant
        python scripts/build_index.py --tenant-id company-a
        
        # Rebuild index from scratch
        python scripts/build_index.py --rebuild
    """
    setup_logging()
    
    console.print("\n[bold blue]PoliRAG Index Builder[/bold blue]\n")
    
    # Show configuration
    config_table = Table(title="Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Index Name", index_name)
    config_table.add_row("Tenant Filter", tenant_id or "All tenants")
    config_table.add_row("Rebuild", str(rebuild))
    console.print(config_table)
    console.print()
    
    asyncio.run(build_index(index_name, tenant_id, rebuild))


async def build_index(index_name: str, tenant_id: str, rebuild: bool):
    """Async index building."""
    try:
        # Initialize components
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing components...", total=None)
            
            doc_manager = DocumentManager()
            chunker = TablePreservingChunker()
            embedding_service = EmbeddingService()
            vector_store = FAISSVectorStore(index_name=index_name)
            bm25_store = BM25Store(index_name=index_name)
            
            progress.update(task, completed=True)
        
        # Load documents
        console.print("[bold]Step 1: Loading documents[/bold]")
        documents = doc_manager.list_documents(tenant_id=tenant_id)
        
        if not documents:
            console.print("[yellow]No documents found[/yellow]")
            return
        
        console.print(f"Found {len(documents)} documents\n")
        
        # Chunk all documents
        console.print("[bold]Step 2: Chunking documents[/bold]")
        all_chunks = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Chunking...", total=len(documents))
            
            for doc in documents:
                chunks = chunker.chunk_document(doc)
                all_chunks.extend(chunks)
                progress.update(task, advance=1)
        
        console.print(f"Created {len(all_chunks)} chunks\n")
        
        # Generate embeddings
        console.print("[bold]Step 3: Generating embeddings[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Embedding...", total=None)
            all_chunks = await embedding_service.embed_chunks(all_chunks, batch_size=100)
            progress.update(task, completed=True)
        
        console.print(f"Generated {len(all_chunks)} embeddings\n")
        
        # Build indexes
        console.print("[bold]Step 4: Building indexes[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task1 = progress.add_task("Building FAISS index...", total=None)
            vector_store.add_chunks(all_chunks)
            progress.update(task1, completed=True)
            
            task2 = progress.add_task("Building BM25 index...", total=None)
            bm25_store.add_chunks(all_chunks)
            progress.update(task2, completed=True)
            
            task3 = progress.add_task("Saving indexes...", total=None)
            vector_store.save()
            bm25_store.save()
            progress.update(task3, completed=True)
        
        console.print("\n[bold green]✓ Index building complete![/bold green]\n")
        
        # Show stats
        stats_table = Table(title="Index Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green", justify="right")
        stats_table.add_row("Total Documents", str(len(documents)))
        stats_table.add_row("Total Chunks", str(len(all_chunks)))
        stats_table.add_row("Vector Index Size", str(vector_store.index.ntotal))
        stats_table.add_row("BM25 Index Size", str(len(bm25_store.chunks)))
        console.print(stats_table)
        
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        raise


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Document ingestion CLI tool."""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking.chunker import TablePreservingChunker
from src.embeddings.service import EmbeddingService
from src.ingestion.pipeline import IngestionPipeline
from src.models import Role
from src.observability.logging import setup_logging
from src.vectorstore.bm25_store import BM25Store
from src.vectorstore.faiss_store import FAISSVectorStore

console = Console()


@click.command()
@click.option(
    "--path",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to file or directory to ingest"
)
@click.option(
    "--tenant-id",
    "-t",
    default="default",
    help="Tenant ID"
)
@click.option(
    "--role",
    "-r",
    type=click.Choice(["admin", "legal", "audit", "compliance", "read_only"]),
    default="read_only",
    multiple=True,
    help="Role scope (can specify multiple)"
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Recursively scan subdirectories"
)
@click.option(
    "--skip-duplicates/--no-skip-duplicates",
    default=True,
    help="Skip duplicate documents"
)
@click.option(
    "--build-index/--no-build-index",
    default=True,
    help="Build vector and BM25 indexes after ingestion"
)
@click.option(
    "--index-name",
    "-i",
    default="main",
    help="Index name"
)
def main(
    path: str,
    tenant_id: str,
    role: tuple,
    recursive: bool,
    skip_duplicates: bool,
    build_index: bool,
    index_name: str
):
    """Ingest documents into PoliRAG.
    
    Examples:
    
        # Ingest a single file
        python scripts/ingest.py --path /path/to/document.pdf --tenant-id company-a --role legal
        
        # Ingest a directory
        python scripts/ingest.py --path /path/to/documents --tenant-id company-a --role legal --role audit
        
        # Ingest without building index (faster for bulk ingestion)
        python scripts/ingest.py --path /path/to/documents --no-build-index
    """
    setup_logging()
    
    console.print("\n[bold blue]PoliRAG Document Ingestion[/bold blue]\n")
    
    # Convert roles
    role_scope = [Role(r) for r in role] if role else [Role.READ_ONLY]
    
    # Show configuration
    config_table = Table(title="Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Path", path)
    config_table.add_row("Tenant ID", tenant_id)
    config_table.add_row("Role Scope", ", ".join(str(r) for r in role_scope))
    config_table.add_row("Recursive", str(recursive))
    config_table.add_row("Skip Duplicates", str(skip_duplicates))
    config_table.add_row("Build Index", str(build_index))
    config_table.add_row("Index Name", index_name)
    console.print(config_table)
    console.print()
    
    # Run ingestion
    asyncio.run(
        ingest(
            path=Path(path),
            tenant_id=tenant_id,
            role_scope=role_scope,
            recursive=recursive,
            skip_duplicates=skip_duplicates,
            build_index=build_index,
            index_name=index_name
        )
    )


async def ingest(
    path: Path,
    tenant_id: str,
    role_scope: List[Role],
    recursive: bool,
    skip_duplicates: bool,
    build_index: bool,
    index_name: str
):
    """Async ingestion function."""
    try:
        # Initialize components
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing components...", total=None)
            
            pipeline = IngestionPipeline()
            chunker = TablePreservingChunker()
            embedding_service = EmbeddingService()
            
            if build_index:
                vector_store = FAISSVectorStore(index_name=index_name)
                bm25_store = BM25Store(index_name=index_name)
            else:
                vector_store = None
                bm25_store = None
            
            progress.update(task, completed=True)
        
        # Ingest documents
        console.print("\n[bold]Step 1: Ingesting documents[/bold]")
        
        if path.is_file():
            results = [
                pipeline.ingest_file(
                    file_path=path,
                    tenant_id=tenant_id,
                    role_scope=role_scope,
                    skip_if_duplicate=skip_duplicates
                )
            ]
        else:
            results = pipeline.ingest_directory(
                directory=path,
                tenant_id=tenant_id,
                role_scope=role_scope,
                recursive=recursive,
                skip_if_duplicate=skip_duplicates
            )
        
        # Show results
        results_table = Table(title="Ingestion Results")
        results_table.add_column("Filename", style="cyan")
        results_table.add_column("Status", style="green")
        results_table.add_column("Time (s)", justify="right")
        
        successful = []
        failed = []
        duplicates = []
        
        for doc, manifest in results:
            status_color = "green" if manifest.status == "completed" else "yellow"
            if manifest.status == "duplicate":
                status_color = "blue"
                duplicates.append((doc, manifest))
            elif manifest.status == "completed":
                successful.append((doc, manifest))
            else:
                failed.append((doc, manifest))
            
            results_table.add_row(
                doc.filename,
                f"[{status_color}]{manifest.status}[/{status_color}]",
                f"{manifest.processing_time_seconds:.2f}"
            )
        
        console.print(results_table)
        console.print(
            f"\nTotal: {len(results)} | "
            f"[green]Success: {len(successful)}[/green] | "
            f"[blue]Duplicates: {len(duplicates)}[/blue] | "
            f"[red]Failed: {len(failed)}[/red]\n"
        )
        
        if not build_index or not successful:
            console.print("[yellow]Skipping index building[/yellow]")
            return
        
        # Build indexes
        console.print("[bold]Step 2: Chunking documents[/bold]")
        all_chunks = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Chunking...", total=len(successful))
            
            for doc, manifest in successful:
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
        
        # Add to indexes
        console.print("[bold]Step 4: Building indexes[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task1 = progress.add_task("Adding to FAISS index...", total=None)
            vector_store.add_chunks(all_chunks)
            progress.update(task1, completed=True)
            
            task2 = progress.add_task("Adding to BM25 index...", total=None)
            bm25_store.add_chunks(all_chunks)
            progress.update(task2, completed=True)
            
            task3 = progress.add_task("Saving indexes...", total=None)
            vector_store.save()
            bm25_store.save()
            progress.update(task3, completed=True)
        
        console.print("\n[bold green]✓ Ingestion complete![/bold green]\n")
        
        # Show final stats
        stats_table = Table(title="Index Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green", justify="right")
        stats_table.add_row("Total Documents", str(len(successful)))
        stats_table.add_row("Total Chunks", str(len(all_chunks)))
        stats_table.add_row("Vector Index Size", str(vector_store.index.ntotal))
        stats_table.add_row("BM25 Index Size", str(len(bm25_store.chunks)))
        console.print(stats_table)
        
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        raise


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Start the PoliRAG API server."""

import sys
from pathlib import Path

import click
import uvicorn
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings

console = Console()


@click.command()
@click.option(
    "--host",
    "-h",
    default=None,
    help="Host to bind to (default: from .env)"
)
@click.option(
    "--port",
    "-p",
    default=None,
    type=int,
    help="Port to bind to (default: from .env)"
)
@click.option(
    "--workers",
    "-w",
    default=None,
    type=int,
    help="Number of workers (default: from .env)"
)
@click.option(
    "--reload/--no-reload",
    default=False,
    help="Enable auto-reload for development"
)
def main(host: str, port: int, workers: int, reload: bool):
    """Start the PoliRAG API server.
    
    Examples:
    
        # Start with default settings
        python scripts/serve.py
        
        # Start on custom host/port
        python scripts/serve.py --host 0.0.0.0 --port 8080
        
        # Start in development mode with auto-reload
        python scripts/serve.py --reload
        
        # Start with 8 workers
        python scripts/serve.py --workers 8
    """
    settings = get_settings()
    
    # Use CLI args or fall back to settings
    host = host or settings.api_host
    port = port or settings.api_port
    workers = workers or settings.api_workers if not reload else 1
    
    console.print("\n[bold blue]Starting PoliRAG API Server[/bold blue]\n")
    console.print(f"Host: [cyan]{host}[/cyan]")
    console.print(f"Port: [cyan]{port}[/cyan]")
    console.print(f"Workers: [cyan]{workers}[/cyan]")
    console.print(f"Reload: [cyan]{reload}[/cyan]")
    console.print(f"\nAPI Documentation: [green]http://{host}:{port}/docs[/green]")
    console.print(f"Metrics: [green]http://{host}:{port}/metrics[/green]\n")
    
    uvicorn.run(
        "src.app.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
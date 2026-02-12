#!/usr/bin/env python3
"""Run evaluation against a golden Q&A dataset (CI-friendly).

Exit code 0 → all thresholds met; exit code 1 → at least one threshold failed.
"""

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.eval.golden_dataset import GoldenDatasetManager
from src.eval.runner import EvaluationRunner
from src.models import Role
from src.observability.logging import setup_logging

console = Console()


@click.command()
@click.option(
    "--dataset",
    "-d",
    default=None,
    type=click.Path(exists=True),
    help="Path to golden_qa.json (default: from .env)",
)
@click.option(
    "--tenant-id",
    "-t",
    default="default",
    help="Tenant ID for evaluation queries",
)
@click.option(
    "--role",
    "-r",
    type=click.Choice(["admin", "legal", "audit", "compliance", "read_only"]),
    default="admin",
    help="User role for evaluation queries",
)
@click.option(
    "--category",
    "-c",
    default=None,
    help="Filter Q&A pairs by category",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Output path for results JSON",
)
@click.option(
    "--index-name",
    "-i",
    default="main",
    help="Index name to evaluate against",
)
@click.option(
    "--create-sample/--no-create-sample",
    default=False,
    help="Create a sample dataset then exit",
)
def main(
    dataset: str,
    tenant_id: str,
    role: str,
    category: str,
    output: str,
    index_name: str,
    create_sample: bool,
):
    """Run evaluation against a golden Q&A dataset.

    Examples:

        # Evaluate with default dataset
        python scripts/evaluate.py

        # Evaluate specific dataset
        python scripts/evaluate.py --dataset data/eval/golden_qa.json

        # Create a sample dataset for bootstrapping
        python scripts/evaluate.py --create-sample
    """
    setup_logging()

    console.print("\n[bold blue]PoliRAG Evaluation Runner[/bold blue]\n")

    settings = get_settings()
    manager = GoldenDatasetManager()

    # --create-sample: seed a dataset and exit
    if create_sample:
        ds = manager.create_sample_dataset()
        path = Path(dataset) if dataset else Path(settings.eval_dataset_path)
        manager.save_dataset(ds, path)
        console.print(
            f"[green]✓ Sample dataset saved to {path} "
            f"({len(ds.qa_pairs)} Q&A pairs)[/green]"
        )
        return

    # Load dataset
    dataset_path = Path(dataset) if dataset else Path(settings.eval_dataset_path)
    if not dataset_path.exists():
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        console.print("[yellow]Tip: run with --create-sample first[/yellow]")
        sys.exit(1)

    golden_ds = manager.load_dataset(dataset_path)
    console.print(
        f"Loaded dataset [cyan]{golden_ds.name}[/cyan] "
        f"with {len(golden_ds.qa_pairs)} Q&A pairs\n"
    )

    user_role = Role(role)

    # Run evaluation
    asyncio.run(
        _evaluate(
            golden_ds=golden_ds,
            index_name=index_name,
            tenant_id=tenant_id,
            user_role=user_role,
            category_filter=category,
            output_path=Path(output) if output else None,
        )
    )


async def _evaluate(golden_ds, index_name, tenant_id, user_role, category_filter, output_path):
    runner = EvaluationRunner(index_name=index_name)

    result = await runner.evaluate_dataset(
        dataset=golden_ds,
        tenant_id=tenant_id,
        user_role=user_role,
        category_filter=category_filter,
    )

    # Display results
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_column("Threshold", style="yellow", justify="right")
    table.add_column("Status", justify="center")

    settings = runner.settings

    rows = [
        ("Recall@5", f"{result.recall_at_k:.3f}", f"≥ {settings.eval_min_recall_at_k}",
         result.recall_at_k >= settings.eval_min_recall_at_k),
        ("MRR", f"{result.mean_reciprocal_rank:.3f}", f"≥ {settings.eval_min_mrr}",
         result.mean_reciprocal_rank >= settings.eval_min_mrr),
        ("Hallucination Rate", f"{result.hallucination_rate:.3f}",
         f"≤ {settings.eval_max_hallucination_rate}",
         result.hallucination_rate <= settings.eval_max_hallucination_rate),
        ("Avg Confidence", f"{result.avg_confidence:.3f}", "—", True),
        ("Avg Latency (ms)", f"{result.avg_latency_ms:.0f}", "—", True),
        ("Total Queries", str(result.total_queries), "—", True),
    ]

    for name, value, threshold, passed in rows:
        status = "[green]✓[/green]" if passed else "[red]✗[/red]"
        table.add_row(name, value, threshold, status)

    console.print(table)

    overall = "[bold green]PASSED[/bold green]" if result.passed else "[bold red]FAILED[/bold red]"
    console.print(f"\nOverall: {overall}\n")

    # Save results
    if output_path:
        runner.save_results(result, output_path)
        console.print(f"Results saved to {output_path}")

    # Exit code for CI
    if not result.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()

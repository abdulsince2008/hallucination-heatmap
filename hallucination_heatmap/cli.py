"""CLI entry point for hallucination heatmap."""

import json
import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console

from .models import Citation, HeatmapResult
from .embedder import Embedder
from .analyzer import analyze_sentences
from .renderer import render_heatmap, render_json

app = typer.Typer(
    name="hallucination-heatmap",
    help="Sentence-level RAG hallucination detection via embedding distance heatmap",
    add_completion=False,
)
console = Console()


@app.command()
def analyze(
    input_file: Path = typer.Argument(..., help="JSON file with generated text and citations"),
    model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", "--model", "-m", help="Sentence transformer model"),
    threshold: float = typer.Option(0.4, "--threshold", "-t", help="Cosine distance threshold for hallucination"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON instead of heatmap"),
    show_chunks: bool = typer.Option(False, "--show-chunks", "-c", help="Show cited chunks in output"),
):
    """Analyze generated text for sentence-level hallucinations."""
    
    # Load input
    try:
        with open(input_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] File not found: {input_file}")
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid JSON: {e}")
        raise typer.Exit(1)
    
    # Validate input
    generated_text = data.get("generated_text", "")
    if not generated_text:
        console.print("[red]Error:[/red] 'generated_text' field is required")
        raise typer.Exit(1)
    
    citations_data = data.get("citations", [])
    citations = []
    for c in citations_data:
        try:
            citations.append(Citation(**c))
        except Exception as e:
            console.print(f"[yellow]Warning:[/red] Skipping invalid citation: {e}")
    
    # Run analysis
    with console.status("[bold green]Loading embedding model...[/bold green]"):
        embedder = Embedder(model_name=model)
    
    with console.status("[bold green]Computing embeddings...[/bold green]"):
        result = analyze_sentences(
            generated_text=generated_text,
            citations=citations,
            embedder=embedder,
            threshold=threshold,
            model_name=model
        )
    
    # Output
    if output_json:
        render_json(result)
    else:
        render_heatmap(result, show_chunks=show_chunks)


@app.command()
def demo():
    """Run a built-in demo with sample data."""
    from .demo import run_demo
    run_demo()


if __name__ == "__main__":
    app()
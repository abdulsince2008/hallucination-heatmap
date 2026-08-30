"""Terminal heatmap renderer using Rich."""

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from .models import HeatmapResult, SentenceAnalysis


console = Console()


def distance_to_color(distance: float, threshold: float) -> str:
    """Map cosine distance to a color name for Rich."""
    # Green (grounded) -> Yellow -> Red (hallucinated)
    if distance <= threshold * 0.5:
        return "green"
    elif distance <= threshold:
        return "yellow"
    elif distance <= threshold * 1.5:
        return "orange3"
    else:
        return "red"


def render_heatmap(result: HeatmapResult, show_chunks: bool = False) -> None:
    """Render the heatmap to terminal."""
    
    # Summary panel
    total = len(result.sentences)
    cited = sum(1 for s in result.sentences if s.citation is not None)
    grounded = sum(1 for s in result.sentences if s.is_grounded)
    hallucinated = sum(1 for s in result.sentences if s.is_grounded is False)
    errors = sum(1 for s in result.sentences if s.error is not None)
    
    summary = Table.grid(padding=1)
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Model:", result.model_name)
    summary.add_row("Threshold:", f"{result.threshold:.2f}")
    summary.add_row("Total sentences:", str(total))
    summary.add_row("Cited sentences:", str(cited))
    summary.add_row("Grounded:", f"[green]{grounded}[/green]")
    summary.add_row("Likely hallucinated:", f"[red]{hallucinated}[/red]")
    summary.add_row("Errors:", f"[yellow]{errors}[/yellow]")
    summary.add_row("Grounded ratio:", f"{result.overall_grounded_ratio:.1%}")
    
    console.print(Panel(summary, title="[bold]Hallucination Heatmap Summary[/bold]", border_style="blue"))
    console.print()
    
    # Detail table
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("#", width=3, justify="right")
    table.add_column("Sentence", min_width=40, max_width=80)
    table.add_column("Distance", width=10, justify="right")
    table.add_column("Status", width=14, justify="center")
    
    if show_chunks:
        table.add_column("Cited Chunk", min_width=30, max_width=60)
    
    for analysis in result.sentences:
        row_num = str(analysis.sentence_idx + 1)
        
        if analysis.error:
            sentence_text = Text(analysis.sentence, style="dim red")
            distance_text = Text("ERROR", style="red")
            status_text = Text("ERROR", style="red")
            chunk_text = Text(analysis.error, style="dim red")
        elif analysis.cosine_distance is None:
            sentence_text = Text(analysis.sentence, style="dim")
            distance_text = Text("N/A", style="dim")
            status_text = Text("NO CITATION", style="dim yellow")
            chunk_text = Text("No citation", style="dim")
        else:
            dist = analysis.cosine_distance
            color = distance_to_color(dist, result.threshold)
            sentence_text = Text(analysis.sentence)
            distance_text = Text(f"{dist:.3f}", style=color)
            
            if analysis.is_grounded:
                status_text = Text("GROUNDED", style="green")
            else:
                status_text = Text("HALLUCINATED", style="red")
            
            if show_chunks and analysis.citation:
                chunk_preview = analysis.citation.chunk_text[:100]
                if len(analysis.citation.chunk_text) > 100:
                    chunk_preview += "..."
                chunk_text = Text(chunk_preview, style="dim cyan")
            else:
                chunk_text = Text("")
        
        if show_chunks:
            table.add_row(row_num, sentence_text, distance_text, status_text, chunk_text)
        else:
            table.add_row(row_num, sentence_text, distance_text, status_text)
    
    console.print(table)
    console.print()
    
    # Legend
    legend = Table.grid(padding=(0, 2))
    legend.add_column()
    legend.add_column()
    legend.add_row("■", "[green]Well grounded (distance ≤ threshold/2)[/green]")
    legend.add_row("■", "[yellow]Moderately grounded (distance ≤ threshold)[/yellow]")
    legend.add_row("■", "[orange3]Weakly grounded (distance ≤ 1.5×threshold)[/orange3]")
    legend.add_row("■", "[red]Likely hallucinated (distance > 1.5×threshold)[/red]")
    
    console.print(Panel(legend, title="[bold]Legend[/bold]", border_style="dim"))


def render_json(result: HeatmapResult) -> None:
    """Output result as JSON."""
    import json
    console.print_json(result.model_dump_json(indent=2))
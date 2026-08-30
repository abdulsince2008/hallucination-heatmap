"""Built-in demo with sample RAG output."""

from .models import Citation
from .embedder import Embedder
from .analyzer import analyze_sentences
from .renderer import render_heatmap


SAMPLE_GENERATED = (
    "The transformer architecture was introduced in the 2017 paper 'Attention Is All You Need' "
    "by Vaswani et al. It replaced recurrence with self-attention mechanisms, enabling parallel "
    "training. The model uses multi-head attention to capture different representation subspaces. "
    "Positional encodings are added to give the model sequence order information. "
    "The original transformer had 6 encoder and 6 decoder layers. "
    "BERT later adapted the encoder-only architecture for pre-training. "
    "GPT uses a decoder-only architecture with causal masking. "
    "Transformers achieve state-of-the-art results on translation benchmarks."
)

SAMPLE_CITATIONS = [
    Citation(
        sentence_idx=0,
        chunk_id="chunk_1",
        chunk_text="The Transformer architecture was proposed in the paper 'Attention Is All You Need' (Vaswani et al., 2017). It introduced self-attention as a replacement for recurrence."
    ),
    Citation(
        sentence_idx=1,
        chunk_id="chunk_2",
        chunk_text="Self-attention allows the model to process all positions in parallel, unlike RNNs which process sequentially."
    ),
    Citation(
        sentence_idx=2,
        chunk_id="chunk_3",
        chunk_text="Multi-head attention projects queries, keys, and values into multiple representation subspaces."
    ),
    Citation(
        sentence_idx=3,
        chunk_id="chunk_4",
        chunk_text="Since self-attention is permutation-invariant, positional encodings are added to inject sequence order."
    ),
    Citation(
        sentence_idx=4,
        chunk_id="chunk_5",
        chunk_text="The base transformer model consists of 6 encoder layers and 6 decoder layers."
    ),
    Citation(
        sentence_idx=5,
        chunk_id="chunk_6",
        chunk_text="BERT (Bidirectional Encoder Representations from Transformers) uses only the encoder stack for masked language modeling."
    ),
    Citation(
        sentence_idx=6,
        chunk_id="chunk_7",
        chunk_text="GPT (Generative Pre-trained Transformer) uses a decoder-only architecture with causal attention masking."
    ),
    # Intentionally wrong citation for sentence 7 (hallucination)
    Citation(
        sentence_idx=7,
        chunk_id="chunk_8",
        chunk_text="Transformers were originally designed for image classification tasks and later adapted to NLP."
    ),
]


def run_demo():
    """Run the demo analysis."""
    from rich.console import Console
    console = Console()
    
    console.print("[bold cyan]Running Hallucination Heatmap Demo[/bold cyan]")
    console.print("=" * 50)
    console.print()
    
    embedder = Embedder()
    
    with console.status("[bold green]Computing embeddings...[/bold green]"):
        result = analyze_sentences(
            generated_text=SAMPLE_GENERATED,
            citations=SAMPLE_CITATIONS,
            embedder=embedder,
            threshold=0.4,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    render_heatmap(result, show_chunks=False)
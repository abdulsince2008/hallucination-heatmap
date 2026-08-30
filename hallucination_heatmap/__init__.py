"""Hallucination Heatmap - Sentence-level RAG hallucination detection."""

from .models import Citation, SentenceAnalysis, HeatmapResult
from .embedder import Embedder
from .analyzer import analyze_sentences
from .renderer import render_heatmap

__all__ = [
    "Citation",
    "SentenceAnalysis",
    "HeatmapResult",
    "Embedder",
    "analyze_sentences",
    "render_heatmap",
]
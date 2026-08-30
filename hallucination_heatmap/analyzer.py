"""Core analysis logic for sentence-level hallucination detection."""

import re
import numpy as np
from typing import List, Optional
from .models import Citation, SentenceAnalysis, HeatmapResult
from .embedder import Embedder


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    # Simple but effective sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def analyze_sentences(
    generated_text: str,
    citations: List[Citation],
    embedder: Embedder,
    threshold: float = 0.3,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> HeatmapResult:
    """
    Analyze each sentence against its cited chunk.
    
    Args:
        generated_text: The full generated text
        citations: List of citations mapping sentence indices to source chunks
        embedder: Embedder instance for computing embeddings
        threshold: Cosine distance threshold (above = likely hallucinated)
        model_name: Name of the embedding model used
    
    Returns:
        HeatmapResult with per-sentence analysis
    """
    sentences = split_sentences(generated_text)
    
    # Build citation lookup by sentence index
    citation_map = {c.sentence_idx: c for c in citations}
    
    # Collect all texts to embed (sentences + cited chunks)
    texts_to_embed = []
    sentence_indices = []
    chunk_indices = []
    
    for idx, sentence in enumerate(sentences):
        citation = citation_map.get(idx)
        if citation:
            texts_to_embed.append(sentence)
            texts_to_embed.append(citation.chunk_text)
            sentence_indices.append(len(texts_to_embed) - 2)
            chunk_indices.append(len(texts_to_embed) - 1)
    
    # Batch embed all texts
    all_embeddings = embedder.embed(texts_to_embed)
    
    # Analyze each sentence
    analyses = []
    grounded_count = 0
    
    for idx, sentence in enumerate(sentences):
        citation = citation_map.get(idx)
        
        if citation is None:
            analyses.append(SentenceAnalysis(
                sentence=sentence,
                sentence_idx=idx,
                error="No citation provided for this sentence"
            ))
            continue
        
        try:
            sent_emb = all_embeddings[sentence_indices[idx]]
            chunk_emb = all_embeddings[chunk_indices[idx]]
            
            distance = embedder.cosine_distance(sent_emb, chunk_emb)
            is_grounded = distance <= threshold
            
            if is_grounded:
                grounded_count += 1
            
            analyses.append(SentenceAnalysis(
                sentence=sentence,
                sentence_idx=idx,
                citation=citation,
                cosine_distance=distance,
                is_grounded=is_grounded
            ))
        except Exception as e:
            analyses.append(SentenceAnalysis(
                sentence=sentence,
                sentence_idx=idx,
                citation=citation,
                error=str(e)
            ))
    
    total_cited = sum(1 for a in analyses if a.citation is not None)
    overall_ratio = grounded_count / total_cited if total_cited > 0 else 0.0
    
    return HeatmapResult(
        sentences=analyses,
        overall_grounded_ratio=overall_ratio,
        model_name=model_name,
        threshold=threshold
    )
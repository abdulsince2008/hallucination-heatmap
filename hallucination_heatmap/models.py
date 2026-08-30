"""Data models for hallucination heatmap."""

from pydantic import BaseModel, Field
from typing import List, Optional


class Citation(BaseModel):
    """A citation linking a sentence to a source chunk."""
    sentence_idx: int = Field(..., description="Index of the sentence in generated text")
    chunk_id: str = Field(..., description="Unique identifier of the source chunk")
    chunk_text: str = Field(..., description="The source text chunk")


class SentenceAnalysis(BaseModel):
    """Analysis result for a single sentence."""
    sentence: str
    sentence_idx: int
    citation: Optional[Citation] = None
    cosine_distance: Optional[float] = None
    is_grounded: Optional[bool] = None
    error: Optional[str] = None


class HeatmapResult(BaseModel):
    """Complete heatmap analysis result."""
    sentences: List[SentenceAnalysis]
    overall_grounded_ratio: float
    model_name: str
    threshold: float
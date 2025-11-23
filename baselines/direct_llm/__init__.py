"""Direct LLM baseline that skips retrieval and answers from model prior only."""

from .runner import DirectLLMRunner

__all__ = ["DirectLLMRunner"]

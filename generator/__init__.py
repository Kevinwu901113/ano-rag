"""Generators for structured notes and answers."""

from .answerer import call_lmstudio
from .note_generator import NoteGenerator

__all__ = ["NoteGenerator", "call_lmstudio"]

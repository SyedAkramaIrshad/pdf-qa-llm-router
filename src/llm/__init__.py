"""LLM module for GLM-4.7 API interactions."""

from .client import GLMClient, get_client
from .schemas import SectionSummary, safe_parse_json, validate_summary
from .prompts import (
    get_metadata_context,
    get_section_summary_prompt,
    get_router_prompt,
    get_error_correction_prompt,
    get_answer_generation_prompt,
    format_sections_for_router,
    get_section_breakdown,
)

__all__ = [
    "GLMClient",
    "get_client",
    "SectionSummary",
    "safe_parse_json",
    "validate_summary",
    "get_metadata_context",
    "get_section_summary_prompt",
    "get_router_prompt",
    "get_error_correction_prompt",
    "get_answer_generation_prompt",
    "format_sections_for_router",
    "get_section_breakdown",
]

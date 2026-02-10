"""Pydantic schemas for structured LLM outputs."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any


class SectionSummary(BaseModel):
    """Schema for PDF section summary output."""

    page_breakdown: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of page ranges and their topics"
    )
    summary: List[str] = Field(description="3-5 bullet points covering main topics")
    keywords: List[str] = Field(description="5-10 important terms, concepts, or entities")
    insights: List[str] = Field(description="Notable data points, conclusions, or observations")

    @field_validator("page_breakdown")
    @classmethod
    def page_breakdown_not_empty(cls, v):
        if not v or len(v) == 0:
            return [{"pages": "unknown", "topic": "No breakdown available"}]
        return v

    @field_validator("summary")
    @classmethod
    def summary_not_empty(cls, v):
        if not v or len(v) == 0:
            return ["No summary available"]
        return v

    @field_validator("keywords")
    @classmethod
    def keywords_not_empty(cls, v):
        if not v or len(v) == 0:
            return ["general"]
        return v

    class Config:
        extra = "allow"


def safe_parse_json(response_text: str) -> Dict[str, Any]:
    """Parse JSON from LLM response with multiple fallback strategies.

    Args:
        response_text: Raw LLM response text

    Returns:
        Parsed dictionary, or empty dict if parsing fails
    """
    import json
    import re

    if not response_text or not response_text.strip():
        return {"summary": ["Empty response"], "keywords": [], "insights": []}

    # Strategy 1: Direct JSON parse
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code blocks
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{.*\}',
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1) if match.lastindex else match.group(0)
                return json.loads(json_str.strip())
            except (json.JSONDecodeError, IndexError):
                continue

    # Strategy 3: Try to find JSON-like structure
    try:
        brace_start = response_text.find('{')
        brace_end = response_text.rfind('}')
        if brace_start >= 0 and brace_end > brace_start:
            json_str = response_text[brace_start:brace_end + 1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Strategy 4: Return safe default
    return {
        "summary": ["Failed to parse LLM response"],
        "keywords": [],
        "insights": []
    }


def validate_summary(data: Dict[str, Any]) -> SectionSummary:
    """Validate and convert dict to SectionSummary with safe defaults.

    Args:
        data: Raw dictionary from LLM response

    Returns:
        Validated SectionSummary instance
    """
    try:
        return SectionSummary(**data)
    except Exception:
        # Return safe default if validation fails
        return SectionSummary(
            summary=data.get("summary", ["Parsing failed"]),
            keywords=data.get("keywords", []),
            insights=data.get("insights", [])
        )

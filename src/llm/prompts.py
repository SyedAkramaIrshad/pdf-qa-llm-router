"""Prompt templates for the PDF QA system."""

from typing import List, Dict, Any


def get_metadata_context(metadata: Dict[str, Any]) -> str:
    """
    Generate metadata context that is injected into ALL LLM prompts.

    This prevents the LLM from predicting invalid page numbers.

    Args:
        metadata: Dictionary with PDF metadata including:
            - total_pages: Total number of pages in PDF
            - total_sections: Total number of sections
            - chunk_size: Pages per section

    Returns:
        Formatted metadata context string
    """
    total_pages = metadata.get("total_pages", 0)
    total_sections = metadata.get("total_sections", 0)
    chunk_size = metadata.get("chunk_size", 10)

    # Build section mapping
    section_lines = []
    for i in range(min(total_sections, 10)):
        start_page = i * chunk_size + 1
        end_page = min((i + 1) * chunk_size, total_pages)
        if i == total_sections - 1:
            end_page = total_pages
        section_lines.append(f"  Section {i + 1}: Pages {start_page}-{end_page}")

    if total_sections > 10:
        section_lines.append("  ...")

    section_mapping = "\n".join(section_lines) if section_lines else "  (No sections)"

    return f"""
PDF METADATA
------------
Total Pages: {total_pages}
Total Sections: {total_sections}
Pages Per Section: {chunk_size}
Valid Page Range: 1 to {total_pages}

SECTION MAPPING:
{section_mapping}

IMPORTANT: Always predict page numbers within the valid range (1 to {total_pages})
"""


def get_section_summary_prompt(
    content: str,
    section_id: int,
    page_start: int,
    page_end: int,
    total_sections: int,
    chunk_size: int
) -> str:
    """
    Generate prompt for summarizing a PDF section.

    Args:
        content: The text content of the section
        section_id: Current section number (1-indexed)
        page_start: Starting page number
        page_end: Ending page number
        total_sections: Total number of sections
        chunk_size: Pages per section

    Returns:
        Formatted prompt string
    """
    return f"""You are analyzing a {chunk_size}-page section (pages {page_start}-{page_end}) from a PDF.

CONTEXT: This is section {section_id} of {total_sections}.

CONTENT:
{content}

Please provide a structured analysis:

1. **SUMMARY**: 3-5 bullet points covering the main topics discussed in this section
2. **KEYWORDS**: 5-10 important terms, concepts, formulas, or entities mentioned
3. **INSIGHTS**: Any notable data points, conclusions, formulas, or unique observations

Respond in JSON format:
{{
    "summary": ["bullet 1", "bullet 2", ...],
    "keywords": ["keyword1", "keyword2", ...],
    "insights": ["insight1", "insight2", ...]
}}"""


def get_router_prompt(
    question: str,
    sections_formatted: str,
    metadata: Dict[str, Any]
) -> str:
    """
    Generate prompt for the page router agent.

    Args:
        question: User's question
        sections_formatted: Formatted section summaries
        metadata: PDF metadata dictionary

    Returns:
        Formatted prompt string
    """
    metadata_context = get_metadata_context(metadata)
    total_pages = metadata.get("total_pages", 0)

    return f"""{metadata_context}

You are a routing agent. Your job is to decide which section of the PDF contains the answer to a question.

QUESTION: {question}

{sections_formatted}

INSTRUCTIONS:
1. Analyze which section summary best matches the question
2. Look at keywords and summary points
3. Output your reasoning AND decision

FORMAT: "Reasoning: [why you chose this section]. Decision: Section X (page Y)"

Your response:"""


def get_error_correction_prompt(
    error_message: str,
    previous_prediction: List[int],
    metadata: Dict[str, Any],
    section_breakdown: str
) -> str:
    """
    Generate prompt for error correction.

    Args:
        error_message: The error message from the tool
        previous_prediction: The LLM's previous (failed) prediction
        metadata: PDF metadata dictionary
        section_breakdown: Formatted section breakdown

    Returns:
        Formatted prompt string
    """
    metadata_context = get_metadata_context(metadata)
    total_pages = metadata.get("total_pages", 0)
    total_sections = metadata.get("total_sections", 0)
    chunk_size = metadata.get("chunk_size", 10)

    return f"""{metadata_context}

⚠️  TOOL ERROR - PLEASE CORRECT

The fetch tool failed with the following error:

ERROR: {error_message}

YOUR PREVIOUS PREDICTION: {previous_prediction}

CONTEXT:
- Valid page range: 1 to {total_pages}
- Total sections: {total_sections}
- Each section covers {chunk_size} pages

SECTION BREAKDOWN:
{section_breakdown}

Please analyze why your prediction failed and provide a corrected prediction.
Consider:
1. Which section actually contains the answer based on summaries?
2. What page range does that section cover?
3. Pick a specific page within that valid range

Your corrected prediction (must be valid, as a list):"""


def get_answer_generation_prompt(
    question: str,
    page_text: str,
    page_numbers: List[int],
    images_block: str = ""
) -> str:
    """
    Generate prompt for answer generation.

    Args:
        question: User's question
        page_text: Text content from fetched pages
        page_numbers: List of page numbers
        images_block: Optional block describing images

    Returns:
        Formatted prompt string
    """
    pages_str = ", ".join(map(str, page_numbers))

    return f"""You are answering a question based on specific PDF pages.

QUESTION: {question}

CONTENT FROM PAGE(S) {pages_str}:
{page_text}

{images_block}

INSTRUCTIONS:
1. Provide a clear, accurate answer using ONLY the information above
2. Include citations: [Page X]
3. If images are provided, incorporate visual information in your answer
4. If the answer isn't in the provided content, say so clearly

Your answer:"""


def format_sections_for_router(sections: List[Dict[str, Any]]) -> str:
    """
    Format section summaries for the router prompt.

    Args:
        sections: List of section dictionaries with summary, keywords, insights

    Returns:
        Formatted string for prompt
    """
    lines = []
    for section in sections:
        section_id = section.get("section_id", 0)
        page_range = section.get("page_range", [0, 0])
        summary = section.get("summary", [])
        keywords = section.get("keywords", [])

        summary_str = "; ".join(summary) if summary else "No summary available"
        keywords_str = ", ".join(keywords) if keywords else "No keywords"

        lines.append(
            f"Section {section_id} (Pages {page_range[0]}-{page_range[1]}):\n"
            f"  Summary: {summary_str}\n"
            f"  Keywords: {keywords_str}"
        )

    return "\n\n".join(lines) if lines else "No sections available"


def get_section_breakdown(metadata: Dict[str, Any]) -> str:
    """
    Generate section breakdown for error correction.

    Args:
        metadata: PDF metadata dictionary

    Returns:
        Formatted section breakdown string
    """
    total_pages = metadata.get("total_pages", 0)
    total_sections = metadata.get("total_sections", 0)
    chunk_size = metadata.get("chunk_size", 10)

    lines = []
    for i in range(total_sections):
        start_page = i * chunk_size + 1
        end_page = min((i + 1) * chunk_size, total_pages)
        if i == total_sections - 1:  # Last section
            end_page = total_pages
        lines.append(f"  Section {i + 1}: Pages {start_page}-{end_page}")

    return "\n".join(lines) if lines else "No sections available"

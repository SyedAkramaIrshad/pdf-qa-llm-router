"""LangGraph implementation for PDF QA system."""

import asyncio
import json
from typing import List, Dict, Any, Optional, TypedDict, Literal
from pathlib import Path

from langgraph.graph import StateGraph, END
from PIL import Image

from ..llm import GLMClient, get_metadata_context, get_section_summary_prompt
from ..llm import get_router_prompt, get_error_correction_prompt, get_answer_generation_prompt
from ..llm import format_sections_for_router, get_section_breakdown
from ..pdf import PDFProcessor
from ..config.settings import get_settings


# CACHE DIRECTORY
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "summaries"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# STATE DEFINITIONS

class PDFQAState(TypedDict):
    """State for the PDF QA workflow."""

    # Input
    question: str
    pdf_path: str

    # PDF metadata
    metadata: Dict[str, Any]

    # Section summaries (from indexing)
    section_summaries: List[Dict[str, Any]]

    # Router output
    predicted_pages: List[int]
    router_confidence: float

    # Fetched content
    fetched_pages: List[int]
    page_texts: Dict[int, str]  # page_number -> text
    page_images: Dict[int, List[Image.Image]]  # page_number -> images

    # Tool errors
    fetch_error: Optional[str]
    retry_count: int

    # Final output
    answer: str
    sources: List[int]


class IndexingState(TypedDict):
    """State for the PDF indexing workflow."""

    pdf_path: str
    metadata: Dict[str, Any]
    section_summaries: List[Dict[str, Any]]
    current_section: int


# PAGE FETCHER TOOL

class PageFetcherTool:
    """Tool for fetching page content from PDF."""

    def __init__(self, processor: Optional[PDFProcessor] = None):
        self.processor = processor or PDFProcessor()

    def fetch_pages(
        self,
        pdf_path: str,
        page_numbers: List[int],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch content from specified pages.

        Args:
            pdf_path: Path to PDF
            page_numbers: List of page numbers to fetch (1-indexed)
            metadata: PDF metadata for validation

        Returns:
            Dictionary with page texts and images
        """
        result = {
            "texts": {},
            "images": {},
            "errors": [],
            "fetched_pages": []
        }

        # Validate page numbers
        total_pages = metadata.get("total_pages", 0)

        for page_num in page_numbers:
            if page_num < 1 or page_num > total_pages:
                result["errors"].append(
                    f"Page {page_num} out of range (1-{total_pages})"
                )
                continue

            try:
                # Extract page content
                content = self.processor.extract_page_content(
                    pdf_path, page_num, include_images=True
                )

                result["texts"][page_num] = content["text"]
                result["fetched_pages"].append(page_num)

                # Store images if present
                if content.get("has_images") and content.get("images"):
                    result["images"][page_num] = [
                        img["image"] for img in content["images"]
                    ]

            except Exception as e:
                result["errors"].append(f"Error fetching page {page_num}: {str(e)}")

        return result

    async def fetch_pages_async(
        self,
        pdf_path: str,
        page_numbers: List[int],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Async version for parallel fetching."""
        # For now, use sync version (file I/O is the bottleneck)
        return self.fetch_pages(pdf_path, page_numbers, metadata)


# Singleton instance
_page_fetcher = PageFetcherTool()


# AGENT NODES

async def summarize_sections_node(state: IndexingState) -> IndexingState:
    """Summarize all PDF sections in parallel."""
    pdf_path = state["pdf_path"]
    metadata = state["metadata"]
    processor = PDFProcessor()
    llm = GLMClient()

    # Get configuration
    settings = get_settings()
    api_delay = settings.api_delay

    total_sections = metadata["total_sections"]
    summaries = []

    print(f"\nSummarizing {total_sections} sections...")

    # Create tasks for parallel processing
    async def summarize_section(section_id: int) -> Dict[str, Any]:
        section_data = processor.extract_section_text(pdf_path, section_id)

        prompt = get_section_summary_prompt(
            content=section_data["full_text"],
            section_id=section_data["section_id"],
            page_start=section_data["page_range"][0],
            page_end=section_data["page_range"][1],
            total_sections=total_sections,
            chunk_size=metadata["chunk_size"]
        )

        try:
            summary_data = await llm.generate_json_async(
                prompt,
                temperature=0.3,
                max_tokens=1000
            )

            return {
                "section_id": section_data["section_id"],
                "page_range": section_data["page_range"],
                "summary": summary_data.get("summary", []),
                "keywords": summary_data.get("keywords", []),
                "insights": summary_data.get("insights", []),
            }
        except Exception as e:
            print(f"⚠️  Section {section_id + 1} summarization failed: {e}")
            return {
                "section_id": section_data["section_id"],
                "page_range": section_data["page_range"],
                "summary": [f"Error: {str(e)}"],
                "keywords": [],
                "insights": [],
            }

    # SEQUENTIAL PROCESSING - one section at a time to avoid rate limits
    # Use parallel only if INDEXING_CONCURRENT is explicitly set > 1
    concurrent_sections = settings.indexing_concurrent

    if concurrent_sections > 1:
        # Parallel processing (use with caution - may hit rate limits)
        for i in range(0, total_sections, concurrent_sections):
            batch = range(i, min(i + concurrent_sections, total_sections))
            batch_results = await asyncio.gather(*[
                summarize_section(section_id) for section_id in batch
            ])
            summaries.extend(batch_results)
            print(f"  ✓ Completed sections {i+1}-{min(i+concurrent_sections, total_sections)}/{total_sections}")
            # Add delay between batches
            if i + concurrent_sections < total_sections:
                await asyncio.sleep(api_delay)
    else:
        # Sequential processing - one at a time (default, avoids rate limits)
        for section_id in range(total_sections):
            result = await summarize_section(section_id)
            summaries.append(result)
            print(f"  ✓ Completed section {section_id + 1}/{total_sections}")
            # Configurable delay between requests
            await asyncio.sleep(api_delay)

    state["section_summaries"] = summaries
    return state


async def router_node(state: PDFQAState) -> PDFQAState:
    """Route question to relevant sections using LLM with reasoning."""
    question = state["question"]
    metadata = state["metadata"]
    sections = state["section_summaries"]

    llm = GLMClient()

    # Format sections for router (configurable cap to control prompt size)
    settings = get_settings()
    max_sections = settings.router_max_sections
    if max_sections and max_sections > 0:
        sections_for_router = sections[:max_sections]
    else:
        sections_for_router = sections

    sections_formatted = format_sections_for_router(sections_for_router)

    # Get router prompt with metadata
    prompt = get_router_prompt(question, sections_formatted, metadata)

    print(f"\nRouting question to relevant sections...")

    try:
        # Ask LLM for reasoning + decision
        response = await llm.generate_text_async(
            prompt,
            temperature=0.3,
            max_tokens=500
        )

        print(f"  → LLM Reasoning:\n{response}")

        # Extract specific page numbers from response
        import re
        chunk_size = metadata["chunk_size"]
        total_pages = metadata["total_pages"]

        predicted_pages = []

        # Try to match "Pages 9-10" or "Pages 9, 10" or "Page 9" or "Decision: Pages X-Y"
        range_match = re.search(r'pages?\s*(\d+)\s*[-–]\s*(\d+)', response.lower())
        list_match = re.search(r'pages?\s*([\d,\s]+)', response.lower())
        single_match = re.search(r'page\s*(\d+)', response.lower())

        if range_match:
            # "Pages 9-10" or "Page 9-10"
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            predicted_pages = list(range(start, end + 1))
        elif list_match:
            # "Pages 9, 10" or "9, 10"
            page_list = list_match.group(1).replace(' ', ',')
            predicted_pages = [int(p.strip()) for p in page_list.split(',') if p.strip().isdigit()]
        elif single_match:
            # "Page 9"
            predicted_pages = [int(single_match.group(1))]

        # Validate pages are in range
        predicted_pages = [p for p in predicted_pages if 1 <= p <= total_pages]

        # Fallback: try to find any reasonable page numbers
        if not predicted_pages:
            numbers = re.findall(r'\b\d+\b', response)
            for num in numbers:
                n = int(num)
                if 1 <= n <= total_pages and n not in predicted_pages:
                    predicted_pages.append(n)

        # If still no pages, use first chunk as fallback
        if not predicted_pages:
            predicted_pages = list(range(1, min(chunk_size + 1, total_pages + 1)))

        # Limit to max 20 pages to avoid too much content
        state["predicted_pages"] = predicted_pages[:20]
        state["router_confidence"] = 0.8

        print(f"  → Selected: Pages {state['predicted_pages']}")

    except Exception as e:
        print(f"  ⚠️  Routing failed: {e}")
        chunk_size = metadata["chunk_size"]
        state["predicted_pages"] = list(range(1, min(chunk_size + 1, metadata["total_pages"] + 1)))
        state["router_confidence"] = 0.1

    return state


async def fetcher_node(state: PDFQAState) -> PDFQAState:
    """Fetch page content using page fetcher tool."""
    pdf_path = state["pdf_path"]
    predicted_pages = state["predicted_pages"]
    metadata = state["metadata"]

    print(f"\nFetching pages {predicted_pages}...")

    result = _page_fetcher.fetch_pages(pdf_path, predicted_pages, metadata)

    state["fetched_pages"] = result["fetched_pages"]
    state["page_texts"] = result["texts"]
    state["page_images"] = result["images"]

    # Check for errors
    if result["errors"]:
        state["fetch_error"] = "; ".join(result["errors"])
        print(f"  ⚠️  Errors: {state['fetch_error']}")
    else:
        state["fetch_error"] = None

    print(f"  ✓ Fetched {len(state['fetched_pages'])} pages")

    return state


async def answer_generator_node(state: PDFQAState) -> PDFQAState:
    """Generate final answer using fetched content."""
    question = state["question"]
    fetched_pages = state["fetched_pages"]
    page_texts = state["page_texts"]
    page_images = state["page_images"]

    llm = GLMClient()

    print(f"\nGenerating answer...")

    # Combine all page texts
    combined_text = "\n\n---\n\n".join([
        f"[Page {p}]\n{page_texts[p]}"
        for p in fetched_pages
    ])

    # Check if we need vision (have images)
    has_images = any(len(imgs) > 0 for imgs in page_images.values())

    if has_images:
        # Use vision for first page with images
        for page_num in fetched_pages:
            if page_num in page_images and page_images[page_num]:
                try:
                    first_image = page_images[page_num][0]

                    # Get text description of image
                    vision_prompt = f"""Question: {question}

Context: This is from page {page_num} of a PDF. Please describe any relevant information in this image that helps answer the question."""

                    vision_response = llm.generate_with_image(
                        vision_prompt,
                        first_image,
                        temperature=0.5,
                        max_tokens=500
                    )

                    # Add vision insight to text
                    combined_text += f"\n\n[Image Analysis from Page {page_num}]\n{vision_response}"
                    print(f"  ✓ Used vision for page {page_num}")
                    break
                except Exception as e:
                    print(f"  ⚠️  Vision failed: {e}")

    # Generate answer
    prompt = get_answer_generation_prompt(
        question, combined_text, fetched_pages
    )

    try:
        answer = await llm.generate_text_async(
            prompt,
            temperature=0.5,
            max_tokens=2000
        )

        state["answer"] = answer
        state["sources"] = fetched_pages

        print(f"  ✓ Answer generated")

    except Exception as e:
        state["answer"] = f"Error generating answer: {str(e)}"
        state["sources"] = fetched_pages

    return state


async def error_correction_node(state: PDFQAState) -> PDFQAState:
    """Correct page prediction based on fetch errors."""
    error_message = state.get("fetch_error")
    previous_prediction = state["predicted_pages"]
    metadata = state["metadata"]
    retry_count = state.get("retry_count", 0)

    if not error_message or retry_count >= 3:
        # No error or max retries reached
        return state

    print(f"\nAttempting error correction (attempt {retry_count + 1})...")

    llm = GLMClient()

    section_breakdown = get_section_breakdown(metadata)

    prompt = get_error_correction_prompt(
        error_message, previous_prediction, metadata, section_breakdown
    )

    try:
        response = await llm.generate_text_async(
            prompt,
            temperature=0.3,
            max_tokens=200
        )

        corrected_pages = parse_page_list(response)

        # Validate
        total_pages = metadata["total_pages"]
        valid_pages = [p for p in corrected_pages if 1 <= p <= total_pages]

        state["predicted_pages"] = valid_pages if valid_pages else [1]
        state["retry_count"] = retry_count + 1
        state["fetch_error"] = None  # Clear error for retry

        print(f"  → Corrected pages: {state['predicted_pages']}")

    except Exception as e:
        print(f"  ⚠️  Error correction failed: {e}")
        state["retry_count"] = retry_count + 1

    return state


# HELPER FUNCTIONS

def parse_page_list(text: str) -> List[int]:
    """Extract page numbers from LLM response.

    Handles formats like:
    - [1, 2, 3]
    - Pages 1, 2, 3
    - 1, 2, 3
    - Page 5
    """
    import re

    # Try to find list pattern [1, 2, 3] or similar
    list_match = re.search(r'\[([\d\s,]+)\]', text)
    if list_match:
        nums = list_match.group(1).split(',')
        return [int(n.strip()) for n in nums if n.strip().isdigit()]

    # Try to extract individual numbers
    numbers = re.findall(r'\b\d+\b', text)
    if numbers:
        return [int(n) for n in numbers if int(n) < 1000]  # Filter out non-page numbers

    return []


# GRAPH BUILDERS

def create_qa_graph() -> StateGraph:
    """Create the QA workflow graph."""

    graph = StateGraph(PDFQAState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("fetcher", fetcher_node)
    graph.add_node("error_correction", error_correction_node)
    graph.add_node("answer_generator", answer_generator_node)

    # Define edges
    graph.set_entry_point("router")

    # Router -> Fetcher
    graph.add_edge("router", "fetcher")

    # Fetcher -> Error Correction (if error) or Answer Generator
    def should_correct(state: PDFQAState) -> Literal["error_correction", "answer_generator"]:
        if state.get("fetch_error") and state.get("retry_count", 0) < 3:
            return "error_correction"
        return "answer_generator"

    graph.add_conditional_edges(
        "fetcher",
        should_correct,
        {
            "error_correction": "error_correction",
            "answer_generator": "answer_generator"
        }
    )

    # Error Correction -> Fetcher (retry)
    graph.add_edge("error_correction", "fetcher")

    # Answer Generator -> END
    graph.add_edge("answer_generator", END)

    return graph.compile()


def create_indexing_graph() -> StateGraph:
    """Create the indexing workflow graph."""

    graph = StateGraph(IndexingState)

    # Add node
    graph.add_node("summarize_sections", summarize_sections_node)

    # Set entry and end
    graph.set_entry_point("summarize_sections")
    graph.add_edge("summarize_sections", END)

    return graph.compile()


# MAIN INTERFACE

class PDFQAAgent:
    """Main agent for PDF QA system."""

    def __init__(self, pdf_path: str):
        """Initialize agent with a PDF.

        Args:
            pdf_path: Path to PDF file
        """
        self.pdf_path = pdf_path
        self.processor = PDFProcessor()
        self.metadata = self.processor.get_pdf_metadata(pdf_path)
        self.qa_graph = create_qa_graph()
        self.indexing_graph = create_indexing_graph()

        # Indexed data
        self._section_summaries: Optional[List[Dict[str, Any]]] = None

    def _get_cache_path(self) -> Path:
        """Get the cache file path for this PDF."""
        import hashlib
        pdf_hash = hashlib.md5(str(self.pdf_path).encode()).hexdigest()
        return CACHE_DIR / f"{pdf_hash}.json"

    def _load_summaries_from_cache(self) -> Optional[List[Dict[str, Any]]]:
        """Load summaries from cache if available."""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    # Verify the PDF hasn't changed
                    if data.get("file_hash") == self.metadata.get("file_hash"):
                        return data.get("summaries", [])
            except Exception as e:
                print(f"  Warning: Failed to load cache: {e}")
        return None

    def _save_summaries_to_cache(self, summaries: List[Dict[str, Any]]) -> None:
        """Save summaries to cache."""
        cache_path = self._get_cache_path()
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    "file_hash": self.metadata.get("file_hash"),
                    "pdf_path": self.pdf_path,
                    "summaries": summaries,
                    "total_sections": len(summaries),
                }, f, indent=2)
        except Exception as e:
            print(f"  Warning: Failed to save cache: {e}")

    async def index_pdf(self, force: bool = False) -> List[Dict[str, Any]]:
        """Index PDF by summarizing all sections.

        Returns:
            List of section summaries
        """
        # Try to load from cache first
        if not force:
            cached = self._load_summaries_from_cache()
            if cached:
                print(f"\nUsing cached summaries ({len(cached)} sections)")
                self._section_summaries = cached
                return cached

        print(f"\n{'='*60}")
        print(f"INDEXING PDF: {Path(self.pdf_path).name}")
        print(f"{'='*60}")

        initial_state: IndexingState = {
            "pdf_path": self.pdf_path,
            "metadata": self.metadata,
            "section_summaries": [],
            "current_section": 0,
        }

        result = await self.indexing_graph.ainvoke(initial_state)

        self._section_summaries = result["section_summaries"]

        # Save to cache
        self._save_summaries_to_cache(self._section_summaries)

        print(f"\nIndexing complete!")
        print(f"   Total Sections: {len(self._section_summaries)}")

        return self._section_summaries

    async def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question about the PDF.

        Args:
            question: User's question

        Returns:
            Dictionary with answer and metadata
        """
        # Ensure PDF is indexed
        if self._section_summaries is None:
            await self.index_pdf()

        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")

        initial_state: PDFQAState = {
            "question": question,
            "pdf_path": self.pdf_path,
            "metadata": self.metadata,
            "section_summaries": self._section_summaries,
            "predicted_pages": [],
            "router_confidence": 0.0,
            "fetched_pages": [],
            "page_texts": {},
            "page_images": {},
            "fetch_error": None,
            "retry_count": 0,
            "answer": "",
            "sources": [],
        }

        result = await self.qa_graph.ainvoke(initial_state)

        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "predicted_pages": result["predicted_pages"],
            "fetched_pages": result["fetched_pages"],
        }


def get_agent(pdf_path: str) -> PDFQAAgent:
    """Get a configured PDF QA agent."""
    return PDFQAAgent(pdf_path)

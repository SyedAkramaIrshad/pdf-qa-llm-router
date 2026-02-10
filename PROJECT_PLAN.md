# Agentic PDF QA System - Project Plan (Updated)

> **Created:** 2026-02-09
> **Project:** 10_feb
> **Framework:** LangGraph
> **LLM:** Z.ai GLM 4.7 (with vision support)

---

## Requirements Summary

| # | Requirement | Implementation |
|---|-------------|----------------|
| 1 | Use GLM 4.7 API (configurable) | `.env` file with API key |
| 2 | Images sent to vision model | GLM-4.7 supports multimodal input via GLM-4V |
| 3 | Error handling via LLM | Feed tool errors back to LLM for self-correction |
| 4 | Metadata awareness | LLM gets page count, section count before tool calls |
| 5 | One PDF at a time | Single PDF processing workflow |

---

## Architecture with Error Correction

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PDF INDEXING (One-time)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   PDF File ──▶ Extract Pages (Text + Images) ──▶ 10-page Sections           │
│                                 │                                            │
│                                 ▼                                            │
│                    10 PARALLEL GLM-4.7 CALLS                                │
│                                 │                                            │
│                                 ▼                                            │
│                    Section Index (JSON)                                      │
│                    {                                                         │
│                      total_pages: 100,                                       │
│                      total_sections: 10,                                     │
│                      sections: [                                             │
│                        {                                                     │
│                          id: 1,                                              │
│                          pages: [1-10],                                      │
│                          summary: "...",                                     │
│                          keywords: ["..."],                                  │
│                          insights: ["..."]                                   │
│                        }                                                     │
│                      ]                                                       │
│                    }                                                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      QUESTION ANSWERING WITH SELF-CORRECTION                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   User Question                                                              │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  ROUTER AGENT (GLM-4.7)                                             │   │
│   │                                                                     │   │
│   │  Input: Question + Metadata + Section Summaries                     │   │
│   │  Metadata: "Total pages: 100, Sections: 10, Range: 1-100"           │   │
│   │  Output: Predicted page numbers [45, 46]                            │   │
│   └─────────────────────────────┬───────────────────────────────────────┘   │
│                                 │                                            │
│                                 ▼                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  FETCH TOOL (Python)                                                │   │
│   │                                                                     │   │
│   │  Input: Page numbers [45, 46]                                       │   │
│   │  Action: Extract text + images from those pages                     │   │
│   │                                                                     │   │
│   │  SUCCESS: {text: "...", images: [base64_data]}                      │   │
│   │  FAILURE: "Error: Page 150 invalid. Valid range: 1-100"             │   │
│   └─────────────────────────────┬───────────────────────────────────────┘   │
│                                 │                                            │
│                    ┌────────────┴────────────┐                              │
│                    ▼                         ▼                              │
│              SUCCESS                    FAILURE                              │
│                    │                         │                              │
│                    ▼                         ▼                              │
│   ┌───────────────────────┐   ┌─────────────────────────────────────────┐   │
│   │  ANSWER AGENT         │   │  ERROR CORRECTION AGENT                  │   │
│   │  (GLM-4.7 + VISION)   │   │  Gets error + metadata                   │   │
│   │                       │   │  "Page 150 invalid. Valid: 1-100."        │   │
│   │  Input: Question +    │   │  "Section 10 covers pages 91-100."        │   │
│   │  Content (text+img)   │   │  Output: Corrected prediction [95]        │   │
│   │  Output: Answer       │   └─────────────────┬───────────────────────┘   │
│   └───────────────────────┘                     │                          │
│                                                    ▼                          │
│                                           Retry fetch with [95]             │
│                                                    │                          │
│                                                    ▼                          │
│                                              Answer Agent                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## LangGraph State with Error Handling

```python
from typing import TypedDict, List, Optional, Annotated
from langgraph.graph import add_messages

class PDFMetadata(TypedDict):
    total_pages: int
    total_sections: int
    chunk_size: int
    filename: str

class SectionInfo(TypedDict):
    section_id: int
    page_range: List[int]  # [1, 10]
    summary: List[str]
    keywords: List[str]
    insights: List[str]

class ToolError(TypedDict):
    tool_name: str
    error_message: str
    valid_range: str
    suggestion: str
    section_info: str  # "Section 10 covers pages 91-100"

class FetchedContent(TypedDict):
    page_number: int
    text: str
    images: List[str]  # Base64 encoded
    has_images: bool

class AgentState(TypedDict):
    # Input
    question: str

    # Metadata (ALWAYS injected into LLM context)
    pdf_metadata: Optional[PDFMetadata]
    section_summaries: Optional[List[SectionInfo]]

    # Router output
    predicted_pages: Optional[List[int]]

    # Tool output
    fetched_content: Optional[List[FetchedContent]]
    tool_error: Optional[ToolError]

    # Retry logic
    retry_count: int
    max_retries: int

    # Final output
    answer: Optional[str]
    reasoning: Optional[str]

    # LangGraph messages
    messages: Annotated[List[str], add_messages]
```

---

## Prompts with Metadata Injection

### Metadata Context (Injected into ALL LLM calls)

```python
def get_metadata_context(metadata: PDFMetadata) -> str:
    """
    This context is ALWAYS included in LLM prompts to prevent
    invalid page predictions.
    """
    return f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                              PDF METADATA                                      ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Total Pages:        {metadata['total_pages']
║  Total Sections:     {metadata['total_sections']
║  Pages Per Section:  {metadata['chunk_size']
║  Valid Page Range:   1 to {metadata['total_pages']
╠═══════════════════════════════════════════════════════════════════════════════╣
║  SECTION MAPPING:                                                            ║
║  Section 1:  Pages 1-{metadata['chunk_size']
║  Section 2:  Pages {metadata['chunk_size']+1}-{metadata['chunk_size']*2}
║  ...                                                                         ║
║  Section {metadata['total_sections']}: Pages {metadata['total_pages']-metadata['chunk_size']+1}-{metadata['total_pages']
╚═══════════════════════════════════════════════════════════════════════════════╝

⚠️  IMPORTANT: Always predict page numbers within the valid range (1 to {total_pages})
"""
```

### Router Prompt with Metadata

```python
PAGE_ROUTER_PROMPT = """
{metadata_context}

You are a page prediction agent. Given a user question and section summaries,
identify which page(s) would contain the answer.

USER QUESTION: {question}

AVAILABLE SECTIONS:
{sections_formatted}

INSTRUCTIONS:
1. First, identify the most relevant section based on summaries and keywords
2. Then, predict the specific page number(s) within that section's range
3. CRITICAL: Page numbers must be between 1 and {total_pages}
4. If unsure, predict multiple pages: [45, 46, 47]

Your prediction (as a list):
"""
```

### Error Correction Prompt

```python
ERROR_CORRECTION_PROMPT = """
{metadata_context}

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

Your corrected prediction (must be valid):
"""
```

### Answer Generation with Vision Support

```python
ANSWER_GENERATION_PROMPT = """
You are answering a question based on specific PDF pages.

QUESTION: {question}

CONTENT FROM PAGE(S) {page_numbers}:
{page_text}

{images_block}

INSTRUCTIONS:
1. Provide a clear, accurate answer using ONLY the information above
2. Include citations: [Page X]
3. If images are provided, incorporate visual information
4. If the answer isn't in the provided content, say so clearly

Your answer:
"""
```

---

## File Structure

```
10_feb/
├── README.md
├── PROJECT_PLAN.md              # This file
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── main.py                    # CLI entry point
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py            # Pydantic config
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py              # GLM-4.7 API client
│   │   └── prompts.py             # All prompt templates
│   │
│   ├── pdf/
│   │   ├── __init__.py
│   │   ├── extractor.py           # PDF text/image extraction
│   │   ├── chunker.py             # 10-page section logic
│   │   └── indexer.py             # Parallel summary generation
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py               # AgentState definition
│   │   ├── graph.py               # LangGraph definition
│   │   │
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── router.py          # Page prediction agent
│   │   │   ├── answer_gen.py      # Answer generation agent (vision)
│   │   │   └── error_corrector.py # Error correction agent
│   │   │
│   │   └── tools/
│   │       ├── __init__.py
│   │       └── content_fetch.py   # Page content fetcher
│   │
│   └── storage/
│       ├── __init__.py
│       ├── index_manager.py       # Load/save index
│       └── pdf_store.py           # Manage PDF data
│
├── data/
│   ├── indices/                   # .json index files
│   ├── pdfs/                      # Original PDFs
│   └── extracted/                 # Extracted page data (pickle)
│
└── tests/
    ├── __init__.py
    ├── test_pdf.py
    ├── test_llm.py
    └── test_agent.py
```

---

## Configuration (.env)

```bash
# Z.ai GLM-4.7 API
GLM_API_KEY=4b6c47a007b044169fd103a26b63e186.Zt3Xo4uRCxWiaU60
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
GLM_MODEL=glm-4.7
GLM_VISION_MODEL=glm-4v

# Processing
CHUNK_SIZE=10                    # Pages per section
MAX_CONCURRENT_CALLS=10          # Parallel LLM calls
MAX_RETRY_ATTEMPTS=3

# Logging
LOG_LEVEL=INFO
```

---

## Requirements

```
# LangGraph & LangChain
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-community>=0.3.0

# PDF Processing
PyPDF2>=3.0.0
pdfplumber>=0.11.0
pdf2image>=1.17.0
Pillow>=11.0.0

# HTTP & Async
httpx>=0.28.0
aiohttp>=3.11.0

# Utils
python-dotenv>=1.0.0
pydantic>=2.0.0
tenacity>=9.0.0
tqdm>=4.67.0
```

---

## Implementation Steps

### Phase 1: Foundation
- [ ] Project structure
- [ ] Configuration system (Pydantic)
- [ ] GLM-4.7 API client
- [ ] .env setup

### Phase 2: PDF Processing
- [ ] PDF text extraction (pdfplumber)
- [ ] PDF image extraction (pdf2image)
- [ ] 10-page chunker
- [ ] Parallel summary generation
- [ ] Index storage

### Phase 3: LangGraph Agent
- [ ] State definition with metadata
- [ ] Router node (with metadata context)
- [ ] Fetch tool (with error returns)
- [ ] Error corrector node
- [ ] Answer node (vision support)

### Phase 4: Integration
- [ ] CLI interface
- [ ] End-to-end testing

---

## Sources

- [ZHIPU AI Official](https://bigmodel.cn/)
- [GLM-4.7 Documentation](https://docs.bigmodel.cn/cn/guide/models/text/glm-4.7)
- [GLM-Image (Vision Model)](https://aihub.caict.ac.cn/models/ZHIPU/GLM-Image)

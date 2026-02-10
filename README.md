# PDF QA System with LLM-Based Hierarchical Routing

An intelligent PDF Question Answering system that uses **LLM-based semantic routing** instead of traditional vector similarity search for precise document navigation.

Scope: Current POC is single-PDF and does not use a vector DB.

## 🎯 Key Innovation

**Current POC (What's Built):**
1. **Ingestion**: Chunk PDF into 10-page sections → Summarize with LLM → Store JSON summaries
2. **Query**: LLM reads JSON summaries → Predicts relevant pages with reasoning
3. **Answer**: Fetch full pages → LLM generates answer with citations

**Why This Works Better Than Traditional RAG:**
- ✅ **No vector database required** - Works with any PDF immediately
- ✅ **Larger context** - 10-page sections preserve content relationships
- ✅ **Smarter routing** - LLM understands semantics vs keyword matching
- ✅ **Explainable** - See LLM's reasoning behind section/page selection
- ✅ **Self-correcting** - Tool errors feed back to LLM for re-routing
- ✅ **Vision support** - Extract and analyze PDF images

## 🚀 Features

- **GLM-4.5/4.6V Flash** - Free models (no billing required)
- **10-page chunking** with sequential LLM summarization
- **LLM-based routing** with explainable reasoning
- **Error correction** - Self-correcting page predictions
- **Vision support** - Extract and analyze PDF images
- **CLI interface** - Simple commands for Q&A
- **Configurable concurrency** - Control parallelism vs rate limits

## 📋 Requirements

```
# Core
langchain-core>=0.2.0
langgraph>=0.2.0
httpx>=0.28.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
tenacity>=9.0.0

# PDF Processing
pypdf>=3.0.0
pdfplumber>=0.11.0
pdf2image>=1.17.0
Pillow>=11.0.0

# CLI
click>=8.0.0
```

## ⚙️ Configuration

```bash
# .env file
GLM_API_KEY=your_api_key_here
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
GLM_MODEL=glm-4.5-flash
GLM_VISION_MODEL=glm-4.6v-flash

CHUNK_SIZE=10              # Pages per section
INDEXING_CONCURRENT=1      # 1=sequential, >1=parallel
API_DELAY=1.0              # Delay between API calls
MAX_CONCURRENT_CALLS=3     # Max parallel LLM calls
```

## 📖 Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env with your GLM API key

# Ask a question
python pdfqa.py ask path/to/document.pdf "What is this about?"

# Interactive mode
python pdfqa.py ask path/to/document.pdf -i

# Index a PDF
python pdfqa.py index path/to/document.pdf

# Show configuration
python pdfqa.py config
```

## 🏗️ Current Architecture (POC)

```
┌─────────────────────────────────────────────────────────────┐
│                    PDF INGESTION PIPELINE                    │
├─────────────────────────────────────────────────────────────┤
│  1. Load single PDF file                                    │
│  2. Chunk into 10-page sections                             │
│  3. Summarize each section with LLM → JSON summaries        │
│  4. Store summaries in memory (for single PDF)              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│  1. User asks question                                      │
│  2. LLM reads JSON summaries → predicts relevant pages      │
│     (with explainable reasoning)                            │
│  3. Fetch full page text + images                           │
│  4. LLM generates answer with citations                     │
│  5. Error correction: Tool failures → LLM re-routes         │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Future Extension: Hybrid Retrieval (Not Implemented)

Future extension (not implemented in this POC): vector-DB-based document discovery
for multi-PDF scale.

```
┌─────────────────────────────────────────────────────────────┐
│                 HYBRID RETRIEVAL ARCHITECTURE               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📁 DOCUMENT DISCOVERY (Coarse Level)                       │
│  ─────────────────────────────────────────                  │
│  1. Embed chunks from all PDFs → Vector DB                 │
│  2. Query → Top 50 chunks (with filename metadata)          │
│  3. Aggregate by filename → Top N unique PDFs               │
│     (e.g., "Which 5 documents are most relevant?")          │
│                                                             │
│  🎯 PRECISE NAVIGATION (Fine Level)                         │
│  ─────────────────────────────────────────                  │
│  4. Load pre-computed JSON summaries for Top N PDFs        │
│  5. Parallel LLM routing (1 call per PDF) → Predict pages   │
│     (Cost scales with N unique files, not fixed)            │
│                                                             │
│  💡 ANSWER GENERATION                                       │
│  ─────────────────────────────────────────                  │
│  6. Fetch pages → LLM generates answer with citations       │
│                                                             │
└─────────────────────────────────────────────────────────────┘

**Why Hybrid?**
- Vector DB is great for document discovery (1000+ PDFs)
- LLM routing is better for precise page navigation within documents
- Combines speed of vector search + semantic understanding of LLM
```

## 📂 Project Structure

```
10_feb/
├── src/
│   ├── agent/          # LangGraph workflow
│   │   └── graph.py    # Router, Fetcher, Answer Generator
│   ├── cli/            # Command-line interface
│   │   └── main.py     # Click commands
│   ├── config/         # Pydantic settings
│   │   └── settings.py  # Configuration with validation
│   ├── llm/            # GLM API client
│   │   ├── client.py    # Text + Vision API calls
│   │   └── prompts.py   # Prompt templates
│   ├── pdf/            # PDF processor
│   │   └── processor.py # Text + image extraction
│   └── storage/        # Vector DB and metadata storage (for scaling)
├── data/
│   ├── pdfs/           # PDF files (not in git)
│   ├── indices/        # Vector DB indices
│   └── summaries/      # Pre-computed JSON summaries
├── pdfqa.py           # Main entry point
├── requirements.txt   # Dependencies
└── .env.example       # Configuration template
```

## 🚀 Extending to Scale

This concept can be extended to handle 1000+ PDFs using a hybrid approach:

**Document Discovery (Vector DB):**
- Embed all PDF chunks into a vector database
- Query returns top relevant chunks with filename metadata
- Aggregate by filename to identify Top N most relevant PDFs

**Precise Navigation (LLM Routing):**
- Load pre-computed JSON summaries for the Top N PDFs
- Run parallel LLM routing (one call per PDF) to predict pages
- Fetch pages and generate answer with citations

**Why Hybrid?**
- Vector DB excels at finding relevant documents at scale
- LLM routing provides semantic understanding within documents
- Cost scales with O(N) where N = unique files in results, not total documents

## 🔐 Security & Privacy

- **No data in repository** - All PDFs and API keys excluded
- **Environment-based config** - Sensitive data in `.env` (gitignored)
- **Rate limiting** - Configurable delays and concurrency
- **Input validation** - Pydantic validates all settings

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For issues and questions, please open an issue on GitHub.

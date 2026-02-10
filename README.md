# PDF QA System with LLM-Based Hierarchical Routing

An intelligent PDF Question Answering system that combines traditional RAG (vector search) with LLM-based routing for precise document navigation.

## 🎯 Key Innovation

**Hybrid Retrieval Architecture:**
1. **Coarse Level**: Vector DB finds relevant chunks → aggregate by filename → top N unique PDFs
2. **Fine Level**: Pre-computed JSON summaries → parallel LLM routing → precise page prediction
3. **Answer Level**: Fetch full pages → LLM generates answer with citations

**Why This Works Better:**
- ✅ **No all-XML parsing required** - Works with any PDF
- ✅ **Larger context** - 10-page sections vs fragmented chunks
- ✅ **Smarter routing** - LLM understands semantics vs keyword matching
- ✅ **Explainable** - See reasoning behind section/page selection
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

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PDF INGESTION PIPELINE                    │
├─────────────────────────────────────────────────────────────┤
│  1. Chunk PDF → Vector DB (with filename metadata)         │
│  2. Summarize sections (10-page chunks) → JSON summaries      │
│  3. Store JSON summaries with metadata                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│  1. Vector Search → Top 50 chunks                             │
│  2. Aggregate by filename → Top N unique PDFs                  │
│  3. Load pre-computed JSON summaries                          │
│  4. Parallel LLM routing (1 per PDF) → Predict pages          │
│  5. Fetch pages → Generate answer with citations              │
└─────────────────────────────────────────────────────────────┘
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

## 🔮 Future Roadmap

### Phase 1: Current (POC)
- ✅ Single PDF Q&A
- ✅ LLM-based routing with reasoning
- ✅ Vision support
- ✅ Error correction
- ✅ CLI interface

### Phase 2: Scaling (100+ PDFs)
- [ ] Vector DB integration (ChunkDB, Weaviate, or pgvector)
- [ ] JSON summary storage
- [ ] Parallel LLM routing (N PDFs simultaneously)
- [ ] Metadata indexing (filename, title, tags)

### Phase 3: Production
- [ ] FastAPI web interface
- [ ] Caching layer (embeddings, summaries)
- [ ] Batch processing pipeline
- [ ] Usage analytics
- [ ] Rate limiting
- [ ] Multi-user support

### Phase 4: Advanced Features
- [ ] Multi-document queries (compare 2+ PDFs)
- [] Conversation memory
- [] File upload API
- [ ] Export to markdown/PDF
- [ ] Citation export

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

---

**Built with ❤️ using GLM-4.5/4.6V Flash (Free models)**

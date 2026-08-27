# Agentic RAG System

An end-to-end Agentic Retrieval-Augmented Generation system that uses autonomous reasoning agents and dynamic vector retrieval to analyze complex, multi-page financial reports and enterprise documents.

---

## Why I Built This

Standard RAG pipelines follow a rigid, linear workflow: take a user question, embed it, search a vector database for top-\$k\$ chunks, and stuff those chunks into an LLM prompt. While this works for simple question answering, it breaks down when dealing with complex, multi-part enterprise documents like annual reports, 10-Ks, and corporate disclosures.

When you ask a question like *"How did company revenue perform across segments, what were the primary operational risks, and how does management's outlook align with their CapEx investments?"*, naive vector retrieval returns scattered, disconnected fragments. The model often hallucinates missing connections because it only gets one shot at retrieval.

I built this project to transition from static retrieval to **Agentic RAG**. Instead of retrieving once upfront, the system equips an autonomous agent (built with CrewAI) with dynamic retrieval tools. The agent actively formulates search queries, reviews retrieved evidence, decides whether more context is needed, and synthesizes a structured, verifiable report grounded strictly in the source document.

---

## What It Does

- **Multi-Format Ingestion**: Ingests multi-page PDFs, Markdown, and text reports with page-level metadata tracking.
- **Context-Preserving Semantic Splitting**: Splits long documents into overlapping chunks (800 chars, 150 overlap) while maintaining paragraph boundaries and section context.
- **Local Zero-Cost Embeddings**: Generates semantic embeddings locally using `sentence-transformers/all-MiniLM-L6-v2` stored in a persistent ChromaDB database.
- **Autonomous Tool-Driven Retrieval**: Exposes vector similarity search as a callable tool for an autonomous agent.
- **Evidence-Grounded Synthesis**: Enforces strict grounding rules where the agent cites exact numbers, metrics, and document pages rather than relying on pre-trained assumptions.
- **Flexible LLM Provider Support**: Replaces hard dependencies on local Ollama daemons with a unified provider layer supporting Groq (free, ultra-fast cloud inference for Llama 3 models), OpenAI, Google Gemini, Anthropic Claude, and local Ollama.

---

## How It Works

```
┌─────────────────┐     ┌───────────────────┐     ┌────────────────────────┐
│ Corporate PDF / │ ──> │  DocumentLoader   │ ──> │    DocumentSplitter    │
│  Annual Report  │     │ (pypdf / text)    │     │ (800 chars, 150 ovlp)  │
└─────────────────┘     └───────────────────┘     └────────────────────────┘
                                                              │
                                                              ▼
┌─────────────────┐     ┌───────────────────┐     ┌────────────────────────┐
│  Agent Output   │ <── │  CrewAI Agent     │ <── │  VectorStoreManager    │
│ (Markdown/CLI)  │     │ (Search Tool Loop)│     │ (ChromaDB + MiniLM-L6) │
└─────────────────┘     └───────────────────┘     └────────────────────────┘
```

1. **Ingest & Chunk**: `DocumentLoader` extracts text and page metadata. `DocumentSplitter` generates overlapping chunks with unique identifiers (`doc_p1_c0`).
2. **Vector Indexing**: Chunks are embedded with `all-MiniLM-L6-v2` and saved into persistent ChromaDB storage (`./company_db`).
3. **Agentic Tool Binding**: Vector search is wrapped as a CrewAI `@tool` (`Financial Report Search`), returning formatted evidence blocks with page references.
4. **Autonomous Reasoning Loop**: The Senior Financial Analyst agent receives the task, decides which queries to execute, evaluates retrieved chunks, and repeats searches if more details are needed.
5. **Structured Output**: The agent outputs a structured markdown report with key metrics, segment breakdowns, risk factors, and management guidance.

---

## Solving the Local Ollama Limitation

The original concept for this workflow relied on a local Ollama daemon (`ollama/llama3.2`). While great for local offline testing, requiring Ollama makes repository reproduction difficult on machines without GPUs, headless cloud servers, Codespaces, or CI pipelines.

To solve this, I designed a **Multi-Provider LLM Factory** (`src/agentic_rag/llm_factory.py`):
- **Groq (Default & Recommended)**: Uses `groq/llama-3.3-70b-versatile` or `groq/llama-3.1-8b-instant`. Runs the exact same open-weight Llama models via cloud inference with zero local RAM/GPU overhead, free-tier access, and sub-second token generation.
- **OpenAI**: Direct integration for `gpt-4o-mini` and `gpt-4o`.
- **Google Gemini**: Integration for `gemini-1.5-flash` and `gemini-2.0-flash`.
- **Anthropic**: Support for `claude-3-5-sonnet`.
- **Local Ollama**: Fully supported as an optional mode (`LLM_PROVIDER=ollama`) for offline environments.

---

## Example Usage & Output

### 1. Ingesting an Annual Report
```bash
python -m agentic_rag.cli ingest data/sample_annual_report.txt --reset
```
Output:
```
Document Ingestion Summary
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Property            ┃ Value                               ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Document Name       │ sample_annual_report.txt            │
│ Raw Pages / Sections│ 1                                   │
│ Chunks Created      │ 5                                   │
│ Total Vectors in DB │ 5                                   │
│ Embedding Model     │ sentence-transformers/all-MiniLM-L6 │
│ Persist Directory   │ /path/to/company_db                 │
└─────────────────────┴─────────────────────────────────────┘
✓ Ingestion completed successfully.
```

### 2. Querying the Agent
```bash
python -m agentic_rag.cli query "What was the total revenue, operating margin, and key growth drivers for FY 2025-2026?"
```
Example Output:
```markdown
### Financial Performance & Growth Breakdown (FY 2025-2026)

Based on the company's annual report, here is the verified performance summary:

- **Consolidated Revenue**: Reached **$28.45 billion**, representing an **8.2% Year-over-Year (YoY)** growth in constant currency terms (up from $26.30B in FY 2024-2025).
- **Operating Margin**: Expanded by **60 basis points to 24.8%**, driven by internal automation initiatives and delivery efficiencies.
- **Net Income**: Recorded at **$6.12 billion**, an increase of **9.4% YoY**.
- **Earnings Per Share (EPS)**: **$4.25** compared to $3.88 in the prior fiscal year.

#### Key Growth Drivers by Vertical:
1. **Healthcare & Life Sciences**: Grew **14.2% YoY** to $3.70 billion.
2. **Digital & Cloud Transformation**: Rose **18.5% YoY**, now accounting for 42% of consolidated revenues.
3. **BFSI (Banking & Financial Services)**: Grew **6.8% YoY** to $8.95 billion.
```

---

## Tech Stack

- **Orchestration & Agents**: [CrewAI](https://github.com/crewAIInc/crewAI)
- **Document Processing**: [LangChain](https://github.com/langchain-ai/langchain), `langchain-text-splitters`, `pypdf`
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) (`langchain-chroma`)
- **Embeddings**: [Sentence-Transformers](https://sbert.net/) (`sentence-transformers/all-MiniLM-L6-v2`)
- **LLM Inference**: [Groq](https://groq.com/), [OpenAI](https://openai.com/), [Google Gemini](https://ai.google.dev/), [Ollama](https://ollama.com/)
- **Configuration & CLI**: Pydantic Settings, Rich, Pytest

---

## Project Structure

```
agentic-rag-system/
├── data/
│   ├── sample_annual_report.txt     # Sample corporate annual report
│   └── annual_report_2025_2026.pdf  # Generated multi-page financial PDF
├── scripts/
│   ├── create_sample_pdf.py         # Utility to generate test PDFs
│   └── run_analysis.py              # Quickstart pipeline demo script
├── src/
│   └── agentic_rag/
│       ├── __init__.py              # Package exports
│       ├── agents.py                # CrewAI agent definitions & backstories
│       ├── cli.py                   # Command-line interface
│       ├── config.py                # Pydantic environment configuration
│       ├── document_loader.py       # PDF/TXT loading & metadata normalization
│       ├── llm_factory.py           # Multi-provider LLM factory (Groq, OpenAI, Gemini, Ollama)
│       ├── pipeline.py              # Main AgenticRAGPipeline coordinator
│       ├── tasks.py                 # Structured analytical task definitions
│       ├── text_splitter.py         # Recursive semantic chunking
│       ├── tools.py                 # Vector search tool for CrewAI
│       └── vector_store.py          # ChromaDB persistence & retrieval
├── tests/
│   ├── test_config.py               # Unit tests for settings and validation
│   ├── test_document_processing.py  # Tests for PDF loading & chunking
│   ├── test_pipeline.py             # Integration tests for pipeline execution
│   ├── test_tools_and_agent.py      # Tests for search tool formatting & tasks
│   └── test_vector_store.py         # Tests for ChromaDB indexing & retrieval
├── .env.example                     # Environment variables template
├── .gitignore                       # Clean Git exclusion rules
├── LICENSE                          # MIT License
├── pyproject.toml                   # Package metadata and dependencies
├── README.md                        # Documentation
└── requirements.txt                 # Frozen dependency list
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/JohnnyWilson-Portfolio/agentic-rag-system.git
cd agentic-rag-system
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
# Or install in editable mode:
pip install -e .
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` and set your preferred provider. For fast free-tier access via Groq:
```env
LLM_PROVIDER=groq
LLM_MODEL=groq/llama-3.3-70b-versatile
GROQ_API_KEY=your_groq_api_key_here
```

*(You can also use `LLM_PROVIDER=openai` with `OPENAI_API_KEY`, `LLM_PROVIDER=gemini` with `GEMINI_API_KEY`, or `LLM_PROVIDER=ollama` if you have Ollama running locally).*

---

## Usage Commands

### Ingest a Document
```bash
# Ingest the included sample report
agentic-rag ingest data/sample_annual_report.txt --reset

# Ingest your own PDF
agentic-rag ingest /path/to/annual_report.pdf --reset
```

### Run an Agentic Query
```bash
agentic-rag query "What are the primary operational risk factors discussed by management?"
```

### Run a Full Financial Audit Report
```bash
agentic-rag analyze --company "Apex Global"
```

### Check System & Vector Store Status
```bash
agentic-rag status
```

---

## Testing

Run the automated test suite with `pytest`:

```bash
pytest tests/ -v
```

The test suite covers:
- Configuration parsing and provider validation.
- Document loading (PDF, text, empty files, format validation).
- Semantic chunking and metadata preservation.
- ChromaDB indexing, embedding generation, and top-\$k\$ similarity search.
- CrewAI retrieval tool execution and evidence formatting.
- End-to-end pipeline coordination and mock agent kickoff.

---

## What I Worked On

- Built the end-to-end Python package architecture connecting document parsing, vector indexing, and agentic workflows.
- Replaced the hard local Ollama dependency with a multi-provider LLM abstraction layer supporting Groq, OpenAI, Google Gemini, and Anthropic.
- Designed custom CrewAI retrieval tools that format retrieved chunks into cited evidence blocks with page numbers.
- Implemented recursive text splitting with enriched metadata tracking (`chunk_id`, `file_name`, `page`).
- Built an interactive CLI with Rich tables and formatted output for document ingestion, targeted queries, and full financial audits.
- Wrote unit and integration tests covering vector store operations, chunking, configuration parsing, and tool execution.

---

## Limitations

- **Complex Tables & Scanned PDFs**: Pure text-based PDF extraction can struggle with multi-column financial tables or scanned image PDFs. Future iterations can integrate OCR pipelines (e.g. `unstructured` or `pdfplumber` table extraction).
- **Single-Document Focus**: The current retriever is optimized for single annual reports or quarterly filings. Cross-document comparative analysis across multiple fiscal years is an area for extension.

---

## What's Next

- [ ] **Tabular Extraction**: Add specialized table parsing using markdown/CSV conversion for complex balance sheets.
- [ ] **Multi-Document Comparison**: Enable side-by-side comparative analysis across multiple years or competitor filings.
- [ ] **Hybrid Search**: Combine BM25 keyword matching with dense vector similarity search for improved domain-specific term retrieval.
- [ ] **Web UI**: Build a lightweight Streamlit or FastHTML interface for drag-and-drop document analysis.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

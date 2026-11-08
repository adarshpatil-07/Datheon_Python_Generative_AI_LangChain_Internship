# Task 2: Document Q&A Assistant with RAG & Vector Embeddings

A Retrieval-Augmented Generation (RAG) system that loads unstructured documents (PDF and TXT), recursively splits them into semantically cohesive chunks, generates vector embeddings using Google Gemini Embeddings, stores them in a persistent ChromaDB vector store with duplicate prevention, and answers questions strictly from the document context with human-readable page citations.

---

## Features

- **Multi-Format Document Ingestion**: Ingests both `.txt` and `.pdf` files from the `documents/` directory.
- **Semantic Recursive Chunking**: Uses `RecursiveCharacterTextSplitter` with section and paragraph boundary awareness (`chunk_size=1000`, `chunk_overlap=200`), ensuring multi-point concepts (such as the three types of machine learning) remain intact.
- **Vector Embeddings**: Powered by Google Generative AI Embeddings (`models/gemini-embedding-001`).
- **Persistent ChromaDB Vector Store**: Persists to `outputs/chroma_db` using deterministic chunk IDs to prevent duplicate chunk re-indexing on repeated executions.
- **Strict Grounded Question Answering**: Answers queries strictly using retrieved document context. If information is absent, returns: `"I couldn't find this information in the provided documents."`
- **Accurate Source Citations**: Provides human-readable document citations and 1-based page citations for PDFs (e.g., `sample_guide.pdf (Page 1)`).

---

## Project Structure

```
Task-2-RAG-Document-QA/
│
├── documents/                     # Input documents directory
│   ├── sample.txt                 # AI and Machine Learning reference document
│   └── sample_guide.pdf           # Sample multi-page PDF document
│
├── outputs/
│   └── chroma_db/                 # Persistent Chroma vector database
│
├── src/
│   ├── document_loader.py         # PDF and TXT loading with metadata extraction
│   ├── text_splitter.py           # Recursive chunking with overlap & section preservation
│   ├── vector_store.py            # ChromaDB initialization with deduplication
│   └── qa.py                      # RAG question answering with strict grounding & citations
│
├── tests/
│   ├── __init__.py
│   └── test_rag.py                # 15 automated unit tests with API mocking
│
├── .env                           # API keys & model configuration (do not commit)
├── requirements.txt               # Project dependencies
├── main.py                        # Interactive CLI assistant
└── README.md                      # Documentation
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 or 3.11
- A Google Gemini API Key

### 2. Activate Virtual Environment & Install Dependencies
From the repository root or project folder:

```powershell
# Navigate to Task 2 directory
cd Task-2-RAG-Document-QA

# If activating the existing venv:
.\venv\Scripts\Activate.ps1

# Or install dependencies into your environment:
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create or verify your `.env` file in `Task-2-RAG-Document-QA/.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

---

## Usage

### Run the Interactive Assistant
Start the interactive CLI:

```powershell
.\venv\Scripts\python.exe main.py
```

### Example Questions & Expected Responses

#### 1. Verifying Full Concept Retrieval (3 Types of Machine Learning)
```
You: What are the three types of machine learning?

Assistant: Based on the provided context, the three types of machine learning are:
1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning

Sources:
- sample.txt
```

#### 2. Querying Multi-Page PDF Documents
```
You: What is discussed on page 1 of the sample guide?

Assistant: Page 1 covers Neural Networks and Deep Learning Architecture.

Sources:
- sample_guide.pdf (Page 1)
```

#### 3. Handling Out-of-Scope Information
```
You: What is quantum computing?

Assistant: I couldn't find this information in the provided documents.
```

Type `exit` or `quit` to exit the assistant.

---

## Running the Automated Test Suite

Run the unit test suite (15 tests covering document loading, chunking, deduplication, citation formatting, and error handling):

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```


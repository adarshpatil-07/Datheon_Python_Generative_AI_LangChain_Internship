# Task 4: Interactive Full-Stack AI Chatbot with Streamlit

A modern, interactive full-stack AI Document Chatbot built with **Streamlit** and powered by **Google Gemini**. It connects directly to the existing RAG pipeline developed in Task 2, maintaining conversation history across the session, streaming answers token-by-token in real time, and displaying human-readable document and PDF page citations.

---

## Features

- **Multi-PDF Upload & Document Processing**: Upload one or multiple PDF documents directly through Streamlit's file uploader. Automatically extracts text page-by-page while preserving exact 0-indexed page metadata.
- **In-Memory Session-Isolated Vector Store**: Each visitor receives a dedicated in-memory ChromaDB vector store collection (`session_<uuid>`). Uploaded documents, chunks, and conversation histories are strictly isolated and never leak or mix between concurrent users.
- **Dynamic Document Management & Removal**: View all active documents with their chunk counts and delete individual files on demand. When a document is removed, its chunks are deleted from the vector store with zero duplicate or stale data left behind.
- **Default Sample Document**: The default sample PDF (`sample_guide.pdf`) is pre-loaded on initial session startup. If removed, it can be re-added instantly via the **"📥 Sample PDF"** button.
- **Real-Time Token Streaming**: Streams responses from Gemini token-by-token directly into the UI using `st.write_stream`.
- **Strict Grounding & Human-Readable Citations**: Answers are strictly grounded in retrieved document chunks across all active PDFs. Displays human-readable document names and 1-based page numbers (e.g., `sample_guide.pdf (Page 1)`). Irrelevant citations are automatically suppressed if the requested information is absent.
- **Robust Error Handling**: Safely handles 0-byte files, corrupted PDFs, scanned image-only documents, missing API keys, and model API errors with friendly UI alerts instead of crashes.

---

## Project Structure

```
Task-4-Streamlit-AI-Chatbot/
│
├── app.py                         # Upgraded Streamlit application with multi-PDF upload
├── requirements.txt               # Streamlit Cloud & local dependencies
├── tests/
│   ├── __init__.py
│   └── test_chatbot.py            # Automated tests for upload, removal, citations & isolation
└── README.md                      # Documentation & execution instructions
```

---

## Setup & Running the Chatbot

### 1. Prerequisites
- Python 3.10 or 3.11
- Task 2 virtual environment (`Task-2-RAG-Document-QA/venv`)
- Task 2 environment configuration (`Task-2-RAG-Document-QA/.env`)

### 2. Launch the Application Locally
Run the Streamlit application using the Task 2 virtual environment from the repository root:

```powershell
.\Task-2-RAG-Document-QA\venv\Scripts\streamlit.exe run Task-4-Streamlit-AI-Chatbot\app.py
```

Or navigate to the `Task-4-Streamlit-AI-Chatbot` directory:

```powershell
cd Task-4-Streamlit-AI-Chatbot
..\Task-2-RAG-Document-QA\venv\Scripts\streamlit.exe run app.py
```

Once started, open your web browser to:
`http://localhost:8501`

---

## Deploying to Streamlit Community Cloud

1. Push your repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io):
   - **Repository**: Select your GitHub repo.
   - **Main file path**: Set to `Task-4-Streamlit-AI-Chatbot/app.py`.
   - **Advanced settings -> Secrets**: Add your Gemini credentials:
     ```toml
     GEMINI_API_KEY = "your_actual_gemini_api_key_here"
     GEMINI_MODEL = "gemini-3.5-flash-lite"
     ```
3. Click **Deploy**. Streamlit Cloud will install dependencies from `Task-4-Streamlit-AI-Chatbot/requirements.txt` automatically.

---

## Running the Automated Test Suite

Run the comprehensive test suite (9 tests covering document extraction, empty/corrupt file handling, multi-document search, deletion, citations, and session isolation):

```powershell
.\Task-2-RAG-Document-QA\venv\Scripts\python.exe -m unittest discover -s Task-4-Streamlit-AI-Chatbot\tests -p "test_*.py" -v
```

---

## Example Queries & Workflow

1. **Querying Across Multiple PDFs**:
   - Upload additional PDFs (e.g., `robotics.pdf`) in the sidebar.
   - Ask: `"Compare machine learning approaches across the documents."`
   - The assistant synthesizes information and cites both documents:
     - `📌 Sources: sample_guide.pdf (Page 1) • robotics.pdf (Page 2)`

2. **Removing a Document**:
   - Click the 🗑️ icon next to any document in the sidebar.
   - The document's chunks are removed from the vector store immediately.
   - Future queries will only retrieve information from remaining active documents.

3. **Grounded Refusal**:
   - Query: `"What is quantum computing?"`
   - Response: `"I couldn't find this information in the provided documents."` (with citations suppressed).

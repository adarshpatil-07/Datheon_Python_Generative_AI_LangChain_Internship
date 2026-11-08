# Task 4: Interactive Full-Stack AI Chatbot with Streamlit

A modern, interactive full-stack AI Document Chatbot built with **Streamlit** and powered by **Google Gemini**. It connects directly to the existing RAG pipeline developed in Task 2, maintaining conversation history across the session, streaming answers token-by-token in real time, and displaying human-readable document and PDF page citations.

---

## Features

- **Full-Stack Web Interface**: Clean, lightweight, responsive UI built with Streamlit (`st.chat_message`, `st.chat_input`, and `st.sidebar`).
- **Real-Time Token Streaming**: Streams responses from Gemini token-by-token directly into the UI using `st.write_stream`.
- **Session State Conversation History**: Preserves the complete dialog context during the session (`st.session_state`).
- **Clear Chat & Reload Options**: One-click conversation reset and RAG cache refresh directly in the sidebar.
- **RAG Backend Reuse**: Reuses the document loader, text splitter, ChromaDB vector store, and Gemini embeddings from `Task-2-RAG-Document-QA` without code duplication.
- **Strict Grounding & Source Citations**: Answers are strictly grounded in retrieved document chunks. Displays human-readable document names and 1-based PDF page numbers (e.g. `sample_guide.pdf (Page 1)`).

---

## Project Structure

```
Task-4-Streamlit-AI-Chatbot/
│
├── app.py                         # Streamlit application entry point
├── tests/
│   ├── __init__.py
│   └── test_chatbot.py            # Unit tests for backend integration & token streaming
└── README.md                      # Documentation & execution instructions
```

---

## Setup & Running the Chatbot

### 1. Prerequisites
- Python 3.10 or 3.11
- Task 2 virtual environment (`Task-2-RAG-Document-QA/venv`)
- Task 2 environment configuration (`Task-2-RAG-Document-QA/.env`)

### 2. Launch the Application
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

## Running the Automated Test Suite

To run the automated unit tests:

```powershell
.\Task-2-RAG-Document-QA\venv\Scripts\python.exe -m unittest discover -s Task-4-Streamlit-AI-Chatbot\tests -p "test_*.py" -v
```

---

## Example Queries

1. **Verify Complete Concept Retrieval**:
   - Query: `"What are the three types of machine learning?"`
   - Streams the 3 types (*Supervised*, *Unsupervised*, and *Reinforcement Learning*) and cites `sample.txt`.

2. **Verify PDF Page Citations**:
   - Query: `"What is discussed on page 1 of the sample guide?"`
   - Streams the answer and cites `sample_guide.pdf (Page 1)`.

3. **Verify Strict Grounding**:
   - Query: `"What is quantum computing?"`
   - Responds: `"I couldn't find this information in the provided documents."` and suppresses irrelevant citations.


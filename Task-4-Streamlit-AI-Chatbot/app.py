import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Resolve Task 2 directory and add to sys.path to reuse existing RAG backend
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
TASK2_DIR = REPO_ROOT / "Task-2-RAG-Document-QA"

if str(TASK2_DIR) not in sys.path:
    sys.path.insert(0, str(TASK2_DIR))

# Load environment variables from Task 2's .env
load_dotenv(TASK2_DIR / ".env")

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.vector_store import create_vector_store
from src.qa import format_citation, MISSING_INFO_MESSAGE


# Page configuration
st.set_page_config(
    page_title="Datheon AI Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def extract_token_text(chunk) -> str:
    """Helper to safely extract text from Gemini message chunks."""
    if hasattr(chunk, "text") and isinstance(chunk.text, str) and chunk.text:
        return chunk.text
    if isinstance(chunk.content, str):
        return chunk.content
    if isinstance(chunk.content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in chunk.content
        )
    return str(chunk.content) if chunk.content else ""


@st.cache_resource(show_spinner="Initializing vector store from documents...")
def get_rag_backend():
    """
    Load and initialize the RAG backend once per session.
    Reuses Task 2's document loader, text splitter, embeddings, and vector store.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from Task-2-RAG-Document-QA/.env")

    documents_dir = TASK2_DIR / "documents"
    persist_dir = TASK2_DIR / "outputs" / "chroma_db"

    # Ingest documents using Task 2 loader
    docs = load_documents(directory=documents_dir)
    if not docs:
        return None, 0, 0, [], api_key, model_name

    # Split documents into semantically coherent chunks
    chunks = split_documents(docs)

    # Initialize Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    # Load or initialize persistent ChromaDB
    vector_store = create_vector_store(
        chunks=chunks,
        embeddings=embeddings,
        persist_directory=persist_dir
    )

    doc_files = [f.name for f in documents_dir.iterdir() if f.is_file() and not f.name.startswith(".")]

    return vector_store, len(docs), len(chunks), doc_files, api_key, model_name


def stream_rag_answer(question: str, vector_store, api_key: str, model_name: str, k: int = 4):
    """
    Retrieve relevant chunks and stream the grounded answer token by token.
    Yields (token, sources). On the first yield, sources are computed.
    """
    # 1. Similarity search in vector store
    try:
        relevant_docs = vector_store.similarity_search(question, k=k)
    except Exception as err:
        raise RuntimeError(f"Vector search failed: {err}") from err

    if not relevant_docs:
        yield MISSING_INFO_MESSAGE, []
        return

    # 2. Build context and format human-readable citations
    context_parts = []
    potential_sources = []

    for doc in relevant_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        citation = format_citation(source, page)

        context_parts.append(f"Source: {citation}\n{doc.page_content}")
        if citation not in potential_sources:
            potential_sources.append(citation)

    context = "\n\n".join(context_parts)
    if not context.strip():
        yield MISSING_INFO_MESSAGE, []
        return

    # 3. Initialize Gemini LLM with streaming support
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a document-based question answering assistant.

Answer the user's question using ONLY the provided context below.

Instructions:
1. Carefully read all the provided context chunks.
2. Answer accurately, completely, and concisely using ONLY the information found in the context.
3. If the question asks for a list (e.g. types, examples), list ALL items mentioned in the context.
4. Do not speculate, extrapolate, or use outside knowledge.
5. If the information is not present or cannot be answered from the provided context, respond EXACTLY with:
   "I couldn't find this information in the provided documents."
6. Never invent facts.

Context:
{context}"""
        ),
        ("human", "{question}")
    ])

    chain = prompt | llm

    # Stream tokens from Gemini
    for chunk in chain.stream({"context": context, "question": question}):
        token = extract_token_text(chunk)
        if token:
            yield token, potential_sources


# --- UI Setup ---

# Sidebar
with st.sidebar:
    st.title("📚 Knowledge Base")
    st.markdown("Datheon Task 4: Interactive RAG Chatbot")
    st.divider()

    try:
        vector_store, doc_count, chunk_count, doc_files, api_key, model_name = get_rag_backend()
        backend_ready = vector_store is not None
    except Exception as init_err:
        backend_ready = False
        st.error(f"⚠️ Initialization Error: {init_err}")
        vector_store, doc_count, chunk_count, doc_files, api_key, model_name = None, 0, 0, [], None, None

    if backend_ready:
        st.success("🟢 RAG Pipeline Active")
        st.metric("Documents Loaded", doc_count)
        st.metric("Total Chunks", chunk_count)
        st.caption(f"🧠 Model: `{model_name}`")
        st.caption("🔍 Embeddings: `models/gemini-embedding-001`")

        if doc_files:
            st.markdown("### 📄 Indexed Documents")
            for f in doc_files:
                st.markdown(f"- `{f}`")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()

    else:
        st.warning("⚠️ No documents loaded or missing credentials.")
        st.info("Ensure `sample.txt` or `.pdf` files exist in `Task-2-RAG-Document-QA/documents/`.")


# Main Chat Interface
st.title("💬 Datheon AI Document Assistant")
st.markdown("Ask questions about your uploaded documents. Answers are strictly grounded in retrieved document chunks with source citations.")

# Initialize conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your AI document assistant. Ask me questions about your uploaded documents, and I'll answer with citations.",
            "sources": []
        }
    ]

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            sources_text = " • ".join(f"`{s}`" for s in msg["sources"])
            st.caption(f"📌 **Sources:** {sources_text}")

# Handle user query
user_query = st.chat_input("Ask a question about your documents (e.g., 'What are the three types of machine learning?')...")

if user_query:
    if not backend_ready:
        st.error("RAG backend is not ready. Please verify your documents and .env configuration.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Assistant response with streaming
        with st.chat_message("assistant"):
            try:
                sources_holder = []

                def generate_tokens():
                    for token, sources in stream_rag_answer(
                        question=user_query,
                        vector_store=vector_store,
                        api_key=api_key,
                        model_name=model_name
                    ):
                        if sources and not sources_holder:
                            sources_holder.extend(sources)
                        yield token

                # Stream response token by token
                full_response = st.write_stream(generate_tokens())

                # If answer indicates missing info, clear sources
                final_sources = []
                if full_response and MISSING_INFO_MESSAGE.lower() not in full_response.lower():
                    final_sources = list(dict.fromkeys(sources_holder))

                if final_sources:
                    sources_text = " • ".join(f"`{s}`" for s in final_sources)
                    st.caption(f"📌 **Sources:** {sources_text}")

                # Save assistant response to conversation history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": final_sources
                })

            except Exception as e:
                error_msg = f"Sorry, an error occurred while processing your question: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })


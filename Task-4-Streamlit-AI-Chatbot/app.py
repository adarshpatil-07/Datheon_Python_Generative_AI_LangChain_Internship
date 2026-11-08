import io
import os
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from pypdf import PdfReader

# Resolve Task 2 directory and add to sys.path to reuse existing RAG backend
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
TASK2_DIR = REPO_ROOT / "Task-2-RAG-Document-QA"

if str(TASK2_DIR) not in sys.path:
    sys.path.insert(0, str(TASK2_DIR))

# Load environment variables from Task 2's .env
load_dotenv(TASK2_DIR / ".env")

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

from src.text_splitter import split_documents
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


@st.cache_resource
def get_embeddings(api_key: str):
    """Cache the embeddings model to avoid re-instantiation across reruns."""
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )


def extract_documents_from_pdf_bytes(file_bytes: bytes, filename: str) -> list[Document]:
    """
    Extract text page-by-page from PDF bytes, preserving 0-indexed page metadata.
    Handles empty files, unreadable pages, and corrupted PDFs gracefully.
    """
    if not file_bytes or len(file_bytes) == 0:
        raise ValueError(f"'{filename}' is an empty file (0 bytes).")

    reader = PdfReader(io.BytesIO(file_bytes))
    if len(reader.pages) == 0:
        raise ValueError(f"'{filename}' has 0 pages.")

    docs = []
    for idx, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(
                    page_content=text.strip(),
                    metadata={"source": filename, "page": idx}
                ))
        except Exception as err:
            st.warning(f"Warning: could not read page {idx + 1} of '{filename}': {err}")

    if not docs:
        raise ValueError(f"No extractable text found in '{filename}'. (The PDF may contain scanned images only).")

    return docs


def load_default_sample_document() -> tuple[str, list[Document]]:
    """Load default sample PDF from Task 2 documents folder."""
    sample_pdf = TASK2_DIR / "documents" / "sample_guide.pdf"
    if sample_pdf.exists():
        bytes_data = sample_pdf.read_bytes()
        return sample_pdf.name, extract_documents_from_pdf_bytes(bytes_data, sample_pdf.name)

    sample_txt = TASK2_DIR / "documents" / "sample.txt"
    if sample_txt.exists():
        text = sample_txt.read_text(encoding="utf-8")
        return sample_txt.name, [Document(page_content=text, metadata={"source": sample_txt.name})]

    return "", []


def stream_rag_answer(question: str, vector_store, api_key: str, model_name: str, k: int = 4):
    """
    Retrieve relevant chunks across all active documents and stream the grounded answer token by token.
    Yields (token, sources). On the first yield, sources are computed.
    """
    # Check if vector store has any documents
    try:
        total_chunks = vector_store._collection.count()
    except Exception:
        total_chunks = 0

    if total_chunks == 0:
        yield MISSING_INFO_MESSAGE, []
        return

    # 1. Similarity search in vector store
    try:
        relevant_docs = vector_store.similarity_search(question, k=min(k, total_chunks))
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
        google_api_key=api_key,
        temperature=0
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


# --- Session State Initialization ---

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your AI document assistant. Upload your PDF files or use the default sample document to ask questions with citations.",
            "sources": []
        }
    ]

if "indexed_documents" not in st.session_state:
    st.session_state.indexed_documents = {}

if "vector_store" not in st.session_state and api_key:
    embeddings = get_embeddings(api_key)
    # Session-isolated in-memory Chroma collection
    st.session_state.vector_store = Chroma(
        collection_name=f"session_{st.session_state.session_id}",
        embedding_function=embeddings
    )
    # Automatically index default sample document on first run
    try:
        sample_name, sample_docs = load_default_sample_document()
        if sample_docs:
            sample_chunks = split_documents(sample_docs)
            st.session_state.vector_store.add_documents(sample_chunks)
            st.session_state.indexed_documents[sample_name] = {
                "chunk_count": len(sample_chunks),
                "type": "sample"
            }
    except Exception as sample_err:
        st.warning(f"Could not load default sample document: {sample_err}")


# Helper to delete an individual document
def delete_document(doc_name: str):
    if "vector_store" in st.session_state and doc_name in st.session_state.indexed_documents:
        try:
            st.session_state.vector_store._collection.delete(where={"source": doc_name})
        except Exception as del_err:
            st.error(f"Error removing chunks for {doc_name}: {del_err}")
        del st.session_state.indexed_documents[doc_name]


# --- Sidebar Interface ---

with st.sidebar:
    st.title("📚 Document Management")
    st.markdown("Datheon Task 4: Multi-PDF RAG Assistant")
    st.divider()

    if not api_key:
        st.error("⚠️ GEMINI_API_KEY is missing from `.env`.")
    else:
        st.caption(f"🧠 Model: `{model_name}`")
        st.caption(f"🔒 Session: `{st.session_state.session_id[:8]}` (Isolated)")

        # PDF Uploader
        st.markdown("### 📤 Upload New PDFs")
        uploaded_files = st.file_uploader(
            "Select one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDF documents to add to your session knowledge base",
            label_visibility="collapsed"
        )

        if uploaded_files:
            newly_added = 0
            for uploaded_file in uploaded_files:
                if uploaded_file.name in st.session_state.indexed_documents:
                    continue  # Already indexed in this session

                try:
                    pdf_bytes = uploaded_file.getvalue()
                    extracted_docs = extract_documents_from_pdf_bytes(pdf_bytes, uploaded_file.name)
                    pdf_chunks = split_documents(extracted_docs)

                    st.session_state.vector_store.add_documents(pdf_chunks)
                    st.session_state.indexed_documents[uploaded_file.name] = {
                        "chunk_count": len(pdf_chunks),
                        "type": "uploaded"
                    }
                    newly_added += 1
                except ValueError as val_err:
                    st.warning(f"Skipped '{uploaded_file.name}': {val_err}")
                except Exception as upload_err:
                    st.error(f"Failed to process '{uploaded_file.name}': {upload_err}")

            if newly_added > 0:
                st.success(f"Successfully processed and indexed {newly_added} document(s)!")

        st.divider()

        # Active Document Management
        st.markdown("### 📑 Active Documents")
        total_chunks = sum(info["chunk_count"] for info in st.session_state.indexed_documents.values())

        if not st.session_state.indexed_documents:
            st.info("No active documents. Upload a PDF or restore the sample document.")
        else:
            st.caption(f"{len(st.session_state.indexed_documents)} document(s) • {total_chunks} total chunks")

            for doc_name, info in list(st.session_state.indexed_documents.items()):
                col_info, col_del = st.columns([0.8, 0.2])
                with col_info:
                    tag = "Sample" if info["type"] == "sample" else "Uploaded"
                    st.markdown(f"📄 **{doc_name}**  \n<small>`{tag} • {info['chunk_count']} chunks`</small>", unsafe_allow_html=True)
                with col_del:
                    if st.button("🗑️", key=f"del_{doc_name}", help=f"Remove '{doc_name}' from session"):
                        delete_document(doc_name)
                        st.rerun()

        st.divider()

        # Action Buttons
        col_restore, col_clear = st.columns(2)
        with col_restore:
            sample_name = "sample_guide.pdf"
            if sample_name not in st.session_state.indexed_documents:
                if st.button("📥 Sample PDF", use_container_width=True, help="Restore default sample PDF"):
                    try:
                        s_name, s_docs = load_default_sample_document()
                        if s_docs:
                            s_chunks = split_documents(s_docs)
                            st.session_state.vector_store.add_documents(s_chunks)
                            st.session_state.indexed_documents[s_name] = {
                                "chunk_count": len(s_chunks),
                                "type": "sample"
                            }
                            st.rerun()
                    except Exception as err:
                        st.error(f"Could not restore sample: {err}")

        with col_clear:
            if st.button("🧹 Clear Chat", use_container_width=True, help="Reset conversation history"):
                st.session_state.messages = []
                st.rerun()


# --- Main Chat Interface ---

st.title("💬 Datheon AI Document Assistant")
st.markdown("Ask questions across all your active PDF documents. Answers are strictly grounded in retrieved context with source and page citations.")

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            sources_text = " • ".join(f"`{s}`" for s in msg["sources"])
            st.caption(f"📌 **Sources:** {sources_text}")

# Handle user query
user_query = st.chat_input("Ask a question across your documents (e.g., 'What are the three types of machine learning?')...")

if user_query:
    if not api_key:
        st.error("API key is not configured. Please set GEMINI_API_KEY in `.env`.")
    elif "vector_store" not in st.session_state or not st.session_state.indexed_documents:
        st.warning("No documents are currently active. Please upload a PDF in the sidebar or restore the sample PDF.")
    else:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate assistant response with token streaming
        with st.chat_message("assistant"):
            try:
                sources_holder = []

                def generate_tokens():
                    for token, sources in stream_rag_answer(
                        question=user_query,
                        vector_store=st.session_state.vector_store,
                        api_key=api_key,
                        model_name=model_name
                    ):
                        if sources and not sources_holder:
                            sources_holder.extend(sources)
                        yield token

                # Stream response token by token
                full_response = st.write_stream(generate_tokens())

                # Finalize sources (suppress citations if information was missing)
                final_sources = []
                if full_response and MISSING_INFO_MESSAGE.lower() not in full_response.lower():
                    final_sources = list(dict.fromkeys(sources_holder))

                if final_sources:
                    sources_text = " • ".join(f"`{s}`" for s in final_sources)
                    st.caption(f"📌 **Sources:** {sources_text}")

                # Save to conversation history
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

from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


MISSING_INFO_MESSAGE = "I couldn't find this information in the provided documents."


def format_citation(source: str, page: int | str | None) -> str:
    """Format human-readable source and 1-based page citation."""
    filename = Path(source).name if source and source != "Unknown" else "Unknown"

    if page is not None:
        try:
            page_num = int(page) + 1
            return f"{filename} (Page {page_num})"
        except (ValueError, TypeError):
            return f"{filename} (Page {page})"

    return filename


def answer_question(question, vector_store, api_key, model_name, k=4):
    """
    Retrieve relevant chunks and generate a factual answer strictly from context.
    Provides human-readable source page citations.
    """
    if not question or not str(question).strip():
        return "Please ask a valid question.", []

    # Retrieve most relevant chunks
    try:
        relevant_docs = vector_store.similarity_search(question, k=k)
    except Exception as error:
        raise RuntimeError(f"Vector search failed: {error}") from error

    if not relevant_docs:
        return MISSING_INFO_MESSAGE, []

    # Build context with citation headers
    context_parts = []
    for doc in relevant_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        source_label = format_citation(source, page)
        context_parts.append(f"Source: {source_label}\n{doc.page_content}")

    context = "\n\n".join(context_parts)

    if not context.strip():
        return MISSING_INFO_MESSAGE, []

    # Initialize Gemini chat model
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

    try:
        response = (prompt | llm).invoke({
            "context": context,
            "question": question
        })
    except Exception as error:
        raise RuntimeError(f"Gemini API request failed: {error}") from error

    answer = response.text if hasattr(response, "text") and response.text else str(response.content)

    # Collect unique source citations (only if information was found)
    source_references = []
    if MISSING_INFO_MESSAGE.lower() not in answer.lower():
        for doc in relevant_docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page")
            citation = format_citation(source, page)
            if citation not in source_references:
                source_references.append(citation)

    return answer, source_references

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Recursively split documents into chunks while preserving metadata.
    Uses section headers and paragraph breaks to maintain semantic coherence.
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    # Filter out empty or whitespace-only chunks
    valid_chunks = [chunk for chunk in chunks if chunk.page_content and chunk.page_content.strip()]

    return valid_chunks

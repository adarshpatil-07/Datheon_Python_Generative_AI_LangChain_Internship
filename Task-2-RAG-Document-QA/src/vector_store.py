import hashlib
from pathlib import Path
from langchain_chroma import Chroma


def _generate_chunk_id(chunk, index: int) -> str:
    """Generate a deterministic, unique ID for a document chunk."""
    source_name = Path(str(chunk.metadata.get("source", "doc"))).stem
    page = str(chunk.metadata.get("page", 0))
    content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:16]
    return f"{source_name}_p{page}_idx{index}_{content_hash}"


def create_vector_store(chunks, embeddings, persist_directory="outputs/chroma_db", force_refresh=False):
    """
    Create or load a persistent Chroma vector store.
    Prevents duplicate indexing by verifying deterministic chunk IDs before adding documents.
    """
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings
    )

    if not chunks:
        return vector_store

    existing = vector_store.get()
    existing_ids = set(existing.get("ids", []))

    # Clean legacy random UUID entries from prior un-deduplicated runs
    is_legacy_uuid_store = existing_ids and all("-" in cid and len(cid) == 36 for cid in existing_ids)
    if force_refresh or is_legacy_uuid_store:
        try:
            vector_store.reset_collection()
            existing_ids = set()
        except Exception:
            pass

    new_chunks = []
    new_ids = []

    for idx, chunk in enumerate(chunks):
        chunk_id = _generate_chunk_id(chunk, idx)
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    if new_chunks:
        vector_store.add_documents(new_chunks, ids=new_ids)

    return vector_store

from langchain_chroma import Chroma


def create_vector_store(chunks, embeddings):
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="outputs/chroma_db"
    )

    return vector_store
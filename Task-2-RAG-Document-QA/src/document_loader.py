from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents(directory="documents"):
    """
    Load PDF and TXT documents from the specified directory.
    Preserves document and page metadata (e.g., PDF page numbers).
    Gracefully skips empty or unreadable files.
    """
    documents = []
    folder = Path(directory)

    if not folder.exists() or not folder.is_dir():
        return documents

    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue

        # Skip empty files
        try:
            if file_path.stat().st_size == 0:
                continue
        except OSError:
            continue

        suffix = file_path.suffix.lower()

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
                loaded = loader.load()
                for doc in loaded:
                    if doc.page_content and doc.page_content.strip():
                        documents.append(doc)

            elif suffix == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
                loaded = loader.load()
                for doc in loaded:
                    if doc.page_content and doc.page_content.strip():
                        documents.append(doc)

        except Exception as error:
            print(f"Warning: Could not load '{file_path.name}': {error}")

    return documents

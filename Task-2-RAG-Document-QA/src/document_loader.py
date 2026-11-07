from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents(directory="documents"):
    documents = []
    folder = Path(directory)

    for file_path in folder.iterdir():
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())

        elif file_path.suffix.lower() == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())

    return documents
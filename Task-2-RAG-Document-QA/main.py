import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.vector_store import create_vector_store
from src.qa import answer_question


# Resolve paths relative to this script
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

if not model_name:
    raise ValueError("GEMINI_MODEL not found in .env")


def main():
    documents_dir = BASE_DIR / "documents"
    persist_dir = BASE_DIR / "outputs" / "chroma_db"

    # Load documents
    docs = load_documents(directory=documents_dir)

    if not docs:
        print(f"No valid documents found in '{documents_dir}'.")
        print("Please add .txt or .pdf files to the documents folder.\n")
        return

    chunks = split_documents(docs)

    print(f"Loaded {len(docs)} document(s)")
    print(f"Created {len(chunks)} chunks")

    # Generate embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    # Create / load vector store
    vector_store = create_vector_store(chunks, embeddings, persist_directory=persist_dir)

    print("\nRAG system is ready!")
    print("Ask questions about your documents.")
    print("Type 'exit' to quit.\n")

    # Interactive question-answering loop
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAssistant: Goodbye!")
            break

        if question.lower() in ["exit", "quit"]:
            print("Assistant: Goodbye!")
            break

        if not question:
            continue

        try:
            answer, sources = answer_question(
                question,
                vector_store,
                api_key,
                model_name
            )

            print("\nAssistant:", answer)

            if sources:
                print("\nSources:")
                for source in sources:
                    print("-", source)

            print("\n" + "-" * 50 + "\n")

        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()

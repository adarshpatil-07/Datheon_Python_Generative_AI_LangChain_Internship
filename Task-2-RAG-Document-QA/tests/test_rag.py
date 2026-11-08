import gc
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure Task-2 directory is on sys.path
TASK2_DIR = Path(__file__).resolve().parent.parent
if str(TASK2_DIR) not in sys.path:
    sys.path.insert(0, str(TASK2_DIR))

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.vector_store import create_vector_store, _generate_chunk_id
from src.qa import answer_question, format_citation, MISSING_INFO_MESSAGE


# Minimal 2-page PDF binary for testing
MINIMAL_2PAGE_PDF = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents 5 0 R /Resources << /Font << /F1 7 0 R >> >> >> endobj
4 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents 6 0 R /Resources << /Font << /F1 7 0 R >> >> >> endobj
5 0 obj << /Length 44 >> stream
BT /F1 12 Tf 50 250 Td (Page 1: Machine Learning Overview) Tj ET
endstream endobj
6 0 obj << /Length 44 >> stream
BT /F1 12 Tf 50 250 Td (Page 2: Deep Learning Applications) Tj ET
endstream endobj
7 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 8
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000224 00000 n 
0000000333 00000 n 
0000000426 00000 n 
0000000519 00000 n 
trailer << /Size 8 /Root 1 0 R >>
startxref
595
%%EOF"""


class FakeEmbeddings:
    """Mock embeddings for testing vector operations without network calls."""

    def __init__(self, dimension=32):
        self.dimension = dimension

    def embed_documents(self, texts):
        return [[0.05 * (i % 10 + 1)] * self.dimension for i, _ in enumerate(texts)]

    def embed_query(self, text):
        return [0.05] * self.dimension


class TestDocumentLoader(unittest.TestCase):
    """Verifies requirement: PDF and TXT loading with metadata preservation."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.dir_path = Path(self.test_dir.name)

    def tearDown(self):
        gc.collect()
        self.test_dir.cleanup()

    def test_load_txt_file(self):
        txt_file = self.dir_path / "sample.txt"
        txt_file.write_text("Hello World from text file.", encoding="utf-8")

        docs = load_documents(self.dir_path)
        self.assertEqual(len(docs), 1)
        self.assertIn("Hello World", docs[0].page_content)
        self.assertEqual(docs[0].metadata.get("page"), None)

    def test_load_pdf_file_preserves_page_metadata(self):
        pdf_file = self.dir_path / "document.pdf"
        pdf_file.write_bytes(MINIMAL_2PAGE_PDF)

        docs = load_documents(self.dir_path)
        self.assertEqual(len(docs), 2)
        # Check page numbers are 0-indexed in raw metadata
        self.assertEqual(docs[0].metadata.get("page"), 0)
        self.assertEqual(docs[1].metadata.get("page"), 1)
        self.assertIn("Page 1", docs[0].page_content)
        self.assertIn("Page 2", docs[1].page_content)

    def test_load_skips_empty_files(self):
        empty_txt = self.dir_path / "empty.txt"
        empty_txt.write_text("", encoding="utf-8")

        docs = load_documents(self.dir_path)
        self.assertEqual(len(docs), 0)

    def test_load_handles_missing_directory(self):
        missing_path = self.dir_path / "does_not_exist"
        docs = load_documents(missing_path)
        self.assertEqual(docs, [])


class TestTextSplitter(unittest.TestCase):
    """Verifies requirement: Recursive chunking & retrieval accuracy preservation."""

    def test_split_documents_preserves_three_types_of_ml(self):
        sample_path = TASK2_DIR / "documents" / "sample.txt"
        if not sample_path.exists():
            self.skipTest("sample.txt not found")

        doc = Document(page_content=sample_path.read_text(encoding="utf-8"), metadata={"source": "sample.txt"})
        chunks = split_documents([doc])

        # Verify that all 3 types of machine learning exist within a single chunk
        chunk_with_types = [
            c for c in chunks
            if "Supervised Learning" in c.page_content
            and "Unsupervised Learning" in c.page_content
            and "Reinforcement Learning" in c.page_content
        ]
        self.assertTrue(
            len(chunk_with_types) >= 1,
            "All three types of machine learning must stay together in at least one chunk to avoid split loss."
        )

    def test_split_documents_preserves_metadata(self):
        doc = Document(page_content="Some content\n\nMore content", metadata={"source": "test.pdf", "page": 2})
        chunks = split_documents([doc], chunk_size=20, chunk_overlap=0)
        for chunk in chunks:
            self.assertEqual(chunk.metadata.get("source"), "test.pdf")
            self.assertEqual(chunk.metadata.get("page"), 2)

    def test_split_empty_documents(self):
        self.assertEqual(split_documents([]), [])
        self.assertEqual(split_documents(None), [])


class TestVectorStore(unittest.TestCase):
    """Verifies requirement: ChromaDB persistence and duplicate prevention."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.persist_dir = self.temp_dir.name
        self.embeddings = FakeEmbeddings()

    def tearDown(self):
        gc.collect()
        self.temp_dir.cleanup()

    def test_deterministic_chunk_id(self):
        doc1 = Document(page_content="Content A", metadata={"source": "doc.txt", "page": 0})
        id1_a = _generate_chunk_id(doc1, 0)
        id1_b = _generate_chunk_id(doc1, 0)
        self.assertEqual(id1_a, id1_b)

        doc2 = Document(page_content="Content B", metadata={"source": "doc.txt", "page": 0})
        id2 = _generate_chunk_id(doc2, 0)
        self.assertNotEqual(id1_a, id2)

    def test_prevent_duplicate_indexing(self):
        chunks = [
            Document(page_content="Machine learning types", metadata={"source": "ml.txt"}),
            Document(page_content="Deep learning networks", metadata={"source": "dl.txt"})
        ]

        # Initial indexing
        vs1 = create_vector_store(chunks, self.embeddings, persist_directory=self.persist_dir)
        count_first = vs1._collection.count()
        self.assertEqual(count_first, 2)

        # Second indexing call with identical chunks
        vs2 = create_vector_store(chunks, self.embeddings, persist_directory=self.persist_dir)
        count_second = vs2._collection.count()
        self.assertEqual(count_second, 2, "Duplicate indexing must be prevented on identical chunks.")

    def test_add_new_chunks_without_duplicating_existing(self):
        chunk1 = Document(page_content="Chunk 1", metadata={"source": "doc1.txt"})
        chunk2 = Document(page_content="Chunk 2", metadata={"source": "doc2.txt"})

        vs = create_vector_store([chunk1], self.embeddings, persist_directory=self.persist_dir)
        self.assertEqual(vs._collection.count(), 1)

        # Add both chunk1 and new chunk2
        vs = create_vector_store([chunk1, chunk2], self.embeddings, persist_directory=self.persist_dir)
        self.assertEqual(vs._collection.count(), 2)


class TestQAPipeline(unittest.TestCase):
    """Verifies requirement: Question answering strictly from context with page citations."""

    def test_format_citation_txt(self):
        citation = format_citation("documents/sample.txt", None)
        self.assertEqual(citation, "sample.txt")

    def test_format_citation_pdf(self):
        # 0-indexed page in metadata becomes 1-indexed for human display
        citation = format_citation("documents/research_paper.pdf", 0)
        self.assertEqual(citation, "research_paper.pdf (Page 1)")

        citation_p3 = format_citation("documents/research_paper.pdf", 2)
        self.assertEqual(citation_p3, "research_paper.pdf (Page 3)")

    def test_answer_question_with_mocked_llm(self):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [
            Document(
                page_content="Supervised, Unsupervised, Reinforcement Learning.",
                metadata={"source": "sample_data/report.pdf", "page": 0}
            )
        ]

        expected_answer = "There are three types: Supervised, Unsupervised, and Reinforcement Learning."

        with patch("src.qa.ChatGoogleGenerativeAI") as mock_chat_cls:
            mock_llm = MagicMock()
            mock_llm.return_value = AIMessage(content=expected_answer)
            mock_llm.invoke.return_value = AIMessage(content=expected_answer)
            mock_chat_cls.return_value = mock_llm

            answer, sources = answer_question(
                question="What are the types of machine learning?",
                vector_store=mock_vs,
                api_key="fake-key",
                model_name="gemini-3.5-flash-lite"
            )

            self.assertIn("Supervised", answer)
            self.assertEqual(sources, ["report.pdf (Page 1)"])

    def test_answer_question_missing_information(self):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [
            Document(page_content="Unrelated text about cooking.", metadata={"source": "cook.txt"})
        ]

        with patch("src.qa.ChatGoogleGenerativeAI") as mock_chat_cls:
            mock_llm = MagicMock()
            mock_llm.return_value = AIMessage(content=MISSING_INFO_MESSAGE)
            mock_llm.invoke.return_value = AIMessage(content=MISSING_INFO_MESSAGE)
            mock_chat_cls.return_value = mock_llm

            answer, sources = answer_question(
                question="What is quantum gravity?",
                vector_store=mock_vs,
                api_key="fake-key",
                model_name="gemini-3.5-flash-lite"
            )

            self.assertEqual(answer, MISSING_INFO_MESSAGE)
            # When missing info, sources list must be empty
            self.assertEqual(sources, [])

    def test_answer_question_empty_docs(self):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []

        answer, sources = answer_question(
            question="Any question",
            vector_store=mock_vs,
            api_key="fake-key",
            model_name="gemini-3.5-flash-lite"
        )

        self.assertEqual(answer, MISSING_INFO_MESSAGE)
        self.assertEqual(sources, [])


if __name__ == "__main__":
    unittest.main()

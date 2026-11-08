import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure Task-4 and Task-2 are on sys.path
TASK4_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TASK4_DIR.parent
TASK2_DIR = REPO_ROOT / "Task-2-RAG-Document-QA"

if str(TASK4_DIR) not in sys.path:
    sys.path.insert(0, str(TASK4_DIR))
if str(TASK2_DIR) not in sys.path:
    sys.path.insert(0, str(TASK2_DIR))

from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk
from langchain_chroma import Chroma

from app import (
    extract_token_text,
    extract_documents_from_pdf_bytes,
    load_default_sample_document,
    stream_rag_answer,
    MISSING_INFO_MESSAGE
)
from src.text_splitter import split_documents
from src.qa import format_citation


MINIMAL_VALID_PDF = b"""%PDF-1.4
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
    """Deterministic mock embeddings for testing vector operations in-memory."""
    def __init__(self, dimension=16):
        self.dimension = dimension

    def embed_documents(self, texts):
        return [[0.1] * self.dimension for _ in texts]

    def embed_query(self, text):
        return [0.1] * self.dimension


class TestStreamlitChatbotUpgrades(unittest.TestCase):
    """Test suite verifying multi-PDF upload, document management, and session isolation."""

    def test_extract_documents_from_valid_pdf(self):
        docs = extract_documents_from_pdf_bytes(MINIMAL_VALID_PDF, "test_doc.pdf")
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["source"], "test_doc.pdf")
        self.assertEqual(docs[0].metadata["page"], 0)
        self.assertIn("Page 1", docs[0].page_content)
        self.assertEqual(docs[1].metadata["page"], 1)
        self.assertIn("Page 2", docs[1].page_content)

    def test_extract_documents_empty_file_raises_error(self):
        with self.assertRaises(ValueError) as ctx:
            extract_documents_from_pdf_bytes(b"", "empty.pdf")
        self.assertIn("empty file", str(ctx.exception).lower())

    def test_extract_documents_corrupt_pdf_raises_error(self):
        with self.assertRaises(Exception):
            extract_documents_from_pdf_bytes(b"corrupt-data-not-a-pdf", "bad.pdf")

    def test_default_sample_document_loads(self):
        name, docs = load_default_sample_document()
        self.assertTrue(bool(name))
        self.assertGreater(len(docs), 0)

    def test_multi_pdf_indexing_and_cross_document_retrieval(self):
        emb = FakeEmbeddings()
        vs = Chroma(collection_name="test_multi_doc", embedding_function=emb)

        # PDF 1
        docs_pdf1 = extract_documents_from_pdf_bytes(MINIMAL_VALID_PDF, "guide_ai.pdf")
        chunks1 = split_documents(docs_pdf1)
        vs.add_documents(chunks1)

        # PDF 2
        docs_pdf2 = [
            Document(page_content="Reinforcement learning in autonomous robotics.", metadata={"source": "robotics.pdf", "page": 0})
        ]
        chunks2 = split_documents(docs_pdf2)
        vs.add_documents(chunks2)

        self.assertEqual(vs._collection.count(), len(chunks1) + len(chunks2))

        # Query across documents
        results = vs.similarity_search("learning", k=4)
        sources = {format_citation(r.metadata.get("source"), r.metadata.get("page")) for r in results}

        self.assertIn("guide_ai.pdf (Page 1)", sources)
        self.assertIn("robotics.pdf (Page 1)", sources)

    def test_document_removal_updates_retrieval(self):
        emb = FakeEmbeddings()
        vs = Chroma(collection_name="test_doc_removal", embedding_function=emb)

        doc_a = [Document(page_content="Alpha topic content.", metadata={"source": "alpha.pdf", "page": 0})]
        doc_b = [Document(page_content="Beta topic content.", metadata={"source": "beta.pdf", "page": 0})]

        vs.add_documents(split_documents(doc_a))
        vs.add_documents(split_documents(doc_b))
        self.assertEqual(vs._collection.count(), 2)

        # Remove alpha.pdf
        vs._collection.delete(where={"source": "alpha.pdf"})
        self.assertEqual(vs._collection.count(), 1)

        # Verify only beta remains
        remaining = vs.similarity_search("content", k=5)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].metadata["source"], "beta.pdf")

    def test_session_isolation(self):
        emb = FakeEmbeddings()
        # Session A
        vs_session_a = Chroma(collection_name="session_user_A", embedding_function=emb)
        vs_session_a.add_documents([
            Document(page_content="Confidential User A data", metadata={"source": "user_a.pdf", "page": 0})
        ])

        # Session B
        vs_session_b = Chroma(collection_name="session_user_B", embedding_function=emb)
        vs_session_b.add_documents([
            Document(page_content="Confidential User B data", metadata={"source": "user_b.pdf", "page": 0})
        ])

        # Verify User A cannot retrieve User B's documents
        res_a = vs_session_a.similarity_search("User", k=5)
        self.assertEqual(len(res_a), 1)
        self.assertEqual(res_a[0].metadata["source"], "user_a.pdf")

        # Verify User B cannot retrieve User A's documents
        res_b = vs_session_b.similarity_search("User", k=5)
        self.assertEqual(len(res_b), 1)
        self.assertEqual(res_b[0].metadata["source"], "user_b.pdf")

    def test_stream_rag_answer_with_mocked_llm(self):
        mock_vs = MagicMock()
        mock_vs._collection.count.return_value = 1
        mock_vs.similarity_search.return_value = [
            Document(
                page_content="Supervised, Unsupervised, and Reinforcement Learning are the three types.",
                metadata={"source": "custom.pdf", "page": 0}
            )
        ]

        mock_chunks = [
            AIMessageChunk(content="The 3 types are: "),
            AIMessageChunk(content="Supervised, Unsupervised, and Reinforcement Learning.")
        ]

        with patch("app.ChatPromptTemplate.from_messages") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.stream.return_value = mock_chunks
            mock_prompt_cls.return_value.__or__.return_value = mock_chain

            stream = stream_rag_answer(
                question="What are the types of machine learning?",
                vector_store=mock_vs,
                api_key="fake-key",
                model_name="gemini-3.5-flash-lite"
            )

            tokens = []
            sources = []
            for token, srcs in stream:
                tokens.append(token)
                if srcs and not sources:
                    sources = srcs

            full_text = "".join(tokens)
            self.assertIn("Supervised", full_text)
            self.assertEqual(sources, ["custom.pdf (Page 1)"])

    def test_stream_rag_answer_empty_store_returns_missing_info(self):
        mock_vs = MagicMock()
        mock_vs._collection.count.return_value = 0

        stream = stream_rag_answer(
            question="Any query",
            vector_store=mock_vs,
            api_key="fake-key",
            model_name="gemini-3.5-flash-lite"
        )

        tokens = []
        sources = []
        for token, srcs in stream:
            tokens.append(token)
            if srcs:
                sources = srcs

        self.assertEqual("".join(tokens), MISSING_INFO_MESSAGE)
        self.assertEqual(sources, [])


if __name__ == "__main__":
    unittest.main()

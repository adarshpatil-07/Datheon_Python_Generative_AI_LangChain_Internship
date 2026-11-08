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
from app import extract_token_text, stream_rag_answer, get_rag_backend, MISSING_INFO_MESSAGE


class TestStreamlitChatbot(unittest.TestCase):
    """Test suite for Task 4 Streamlit RAG Chatbot."""

    def test_app_file_exists(self):
        app_path = TASK4_DIR / "app.py"
        self.assertTrue(app_path.exists())
        self.assertGreater(app_path.stat().st_size, 0)

    def test_extract_token_text_string(self):
        chunk = AIMessageChunk(content="Hello")
        self.assertEqual(extract_token_text(chunk), "Hello")

    def test_extract_token_text_list_of_dicts(self):
        chunk = AIMessageChunk(content=[{"type": "text", "text": "Token1"}, {"type": "text", "text": "Token2"}])
        self.assertEqual(extract_token_text(chunk), "Token1Token2")

    def test_extract_token_text_empty(self):
        chunk = AIMessageChunk(content=[])
        self.assertEqual(extract_token_text(chunk), "")

    def test_backend_initialization(self):
        vector_store, doc_count, chunk_count, doc_files, api_key, model_name = get_rag_backend()
        self.assertIsNotNone(vector_store)
        self.assertGreater(doc_count, 0)
        self.assertGreater(chunk_count, 0)
        self.assertIn("sample.txt", doc_files)
        self.assertTrue(bool(api_key))
        self.assertTrue(bool(model_name))

    def test_stream_rag_answer_with_mocked_llm(self):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [
            Document(
                page_content="Supervised, Unsupervised, and Reinforcement Learning are the 3 types.",
                metadata={"source": "sample_guide.pdf", "page": 0}
            )
        ]

        mock_chunks = [
            AIMessageChunk(content="The 3 types are "),
            AIMessageChunk(content="Supervised, Unsupervised, "),
            AIMessageChunk(content="and Reinforcement Learning.")
        ]

        with patch("app.ChatPromptTemplate.from_messages") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.stream.return_value = mock_chunks
            mock_prompt_cls.return_value.__or__.return_value = mock_chain

            stream = stream_rag_answer(
                question="What are the three types of machine learning?",
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
            self.assertEqual(sources, ["sample_guide.pdf (Page 1)"])

    def test_stream_rag_answer_missing_info_empty_docs(self):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []

        stream = stream_rag_answer(
            question="What is quantum entanglement?",
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

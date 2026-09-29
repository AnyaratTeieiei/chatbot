"""Offline checks for the maintained Gemini SDK adapter."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gemini_service


class GeminiServiceTests(unittest.TestCase):
    def test_empty_key_does_not_create_client(self):
        with patch("gemini_service.genai.Client") as client:
            self.assertIsNone(gemini_service.create_client("  "))
            client.assert_not_called()

    def test_client_uses_bounded_exponential_retry_policy(self):
        with patch("gemini_service.genai.Client") as client_factory:
            gemini_service.create_client("test-key")
        options = client_factory.call_args.kwargs["http_options"]
        self.assertEqual(options.timeout, 45_000)
        self.assertEqual(options.retry_options.attempts, 3)
        self.assertEqual(options.retry_options.initial_delay, 1)
        self.assertEqual(options.retry_options.max_delay, 4)

    def test_embedding_adapter_handles_single_and_batch(self):
        client = Mock()
        client.models.embed_content.side_effect = [
            SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])]),
            SimpleNamespace(embeddings=[
                SimpleNamespace(values=[0.1, 0.2]),
                SimpleNamespace(values=[0.3, 0.4]),
            ]),
        ]
        embed = gemini_service.embedding_function(client)

        single = embed(model="gemini-embedding-001", content="คำถาม",
                       task_type="retrieval_query")
        batch = embed(model="gemini-embedding-001", content=["หนึ่ง", "สอง"],
                      task_type="retrieval_document")

        self.assertEqual(single["embedding"], [0.1, 0.2])
        self.assertEqual(batch["embedding"], [[0.1, 0.2], [0.3, 0.4]])
        first_config = client.models.embed_content.call_args_list[0].kwargs["config"]
        self.assertEqual(str(first_config.task_type), "RETRIEVAL_QUERY")

    def test_embedding_adapter_rejects_wrong_count(self):
        client = Mock()
        client.models.embed_content.return_value = SimpleNamespace(embeddings=[])
        with self.assertRaises(ValueError):
            gemini_service.embedding_function(client)(
                model="gemini-embedding-001", content="คำถาม"
            )


if __name__ == "__main__":
    unittest.main()

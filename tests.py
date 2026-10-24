"""Automated test suite for Study Assistant AI.

Tests:
- Calculator safe arithmetic and boundary enforcement
- Study notes accurate retrieval and false-match prevention
- API configuration and missing-key handling
- API error classification and user-friendly diagnostics
- Agent execution with mocked Gemini client (no real API key needed)
"""

import math
import os
import unittest
from unittest.mock import MagicMock, patch

from app import (
    NOTES,
    ask_agent,
    calculator,
    format_api_error,
    get_client,
    search_notes,
)


class TestCalculator(unittest.TestCase):
    """Unit tests for AST-based safe calculator tool."""

    def test_basic_addition(self):
        self.assertEqual(calculator("25 + 15"), "40")

    def test_basic_subtraction(self):
        self.assertEqual(calculator("100 - 37"), "63")

    def test_basic_multiplication(self):
        self.assertEqual(calculator("25 * 16"), "400")

    def test_basic_division(self):
        self.assertEqual(calculator("100 / 4"), "25.0")

    def test_modulo(self):
        self.assertEqual(calculator("17 % 5"), "2")

    def test_exponentiation(self):
        self.assertEqual(calculator("2 ** 8"), "256")

    def test_unary_operators(self):
        self.assertEqual(calculator("-15 + +5"), "-10")
        self.assertEqual(calculator("--5"), "5")

    def test_compound_expression(self):
        self.assertEqual(calculator("(10 + 5) * 2 - 4 / 2"), "28.0")

    def test_division_by_zero(self):
        res = calculator("100 / 0")
        self.assertTrue(res.startswith("ERROR:"))
        self.assertIn("Division by zero", res)

    def test_modulo_by_zero(self):
        res = calculator("50 % 0")
        self.assertTrue(res.startswith("ERROR:"))
        self.assertIn("Division by zero", res)

    def test_exponent_too_large(self):
        res = calculator("2 ** 500")
        self.assertTrue(res.startswith("ERROR:"))
        self.assertIn("Exponent or calculation too large", res)

    def test_oversized_base_exponent(self):
        res = calculator("5000 ** 20")
        self.assertTrue(res.startswith("ERROR:"))
        self.assertIn("Exponent or calculation too large", res)

    def test_expression_too_long(self):
        long_expr = "1 + " * 60 + "1"
        res = calculator(long_expr)
        self.assertEqual(res, "ERROR: Expression is too long.")

    def test_empty_expression(self):
        res = calculator("   ")
        self.assertEqual(res, "ERROR: Empty expression.")

    def test_invalid_syntax(self):
        res = calculator("25 ++* 16")
        self.assertTrue(res.startswith("ERROR:"))

    def test_rejects_import_injection(self):
        res = calculator("__import__('os').system('whoami')")
        self.assertTrue(res.startswith("ERROR:"))
        self.assertNotIn("whoami", res)

    def test_rejects_variable_names(self):
        res = calculator("x + 5")
        self.assertTrue(res.startswith("ERROR:"))

    def test_rejects_function_calls(self):
        res = calculator("eval('10 + 10')")
        self.assertTrue(res.startswith("ERROR:"))

    def test_rejects_file_access(self):
        res = calculator("open('app.py').read()")
        self.assertTrue(res.startswith("ERROR:"))


class TestSearchNotes(unittest.TestCase):
    """Unit tests for study-notes retrieval and false-match prevention."""

    def test_all_five_topics_present_in_knowledge_base(self):
        expected_topics = [
            "supervised learning",
            "unsupervised learning",
            "overfitting",
            "classification",
            "regression",
        ]
        for topic in expected_topics:
            self.assertIn(topic, NOTES)
            self.assertTrue(len(NOTES[topic]) > 10)

    def test_retrieves_supervised_learning(self):
        res = search_notes("What is supervised learning?")
        self.assertIn("Topic: supervised learning", res)
        self.assertIn("labeled examples", res)

    def test_retrieves_unsupervised_learning(self):
        res = search_notes("Explain unsupervised learning")
        self.assertIn("Topic: unsupervised learning", res)
        self.assertIn("unlabeled data", res)

    def test_retrieves_overfitting(self):
        res = search_notes("Tell me about overfitting")
        self.assertIn("Topic: overfitting", res)
        self.assertIn("fits training data too closely", res)

    def test_retrieves_classification(self):
        res = search_notes("Define classification")
        self.assertIn("Topic: classification", res)
        self.assertIn("predicts a category or class", res)

    def test_retrieves_regression(self):
        res = search_notes("What is regression?")
        self.assertIn("Topic: regression", res)
        self.assertIn("continuous numerical value", res)

    def test_unsupported_topic_returns_no_match(self):
        res = search_notes("What is quantum computing?")
        self.assertTrue(res.startswith("NO_MATCH:"))

    def test_another_unsupported_topic(self):
        res = search_notes("Tell me about blockchain smart contracts")
        self.assertTrue(res.startswith("NO_MATCH:"))

    def test_avoids_false_match_on_generic_learning(self):
        # 'learning' alone is generic and must not falsely return supervised learning
        res = search_notes("What is deep learning?")
        self.assertTrue(res.startswith("NO_MATCH:"))

    def test_avoids_false_match_on_machine_learning(self):
        res = search_notes("What is machine learning?")
        self.assertTrue(res.startswith("NO_MATCH:"))

    def test_empty_query(self):
        res = search_notes("   ")
        self.assertTrue(res.startswith("NO_MATCH:"))


class TestConfigurationAndSecurity(unittest.TestCase):
    """Tests for environment variables, credentials handling, and client setup."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=True)
    def test_missing_api_key_returns_none(self):
        client = get_client()
        self.assertIsNone(client)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_test_key", "GEMINI_TIMEOUT_SECONDS": "45"}, clear=True)
    @patch("app.genai.Client")
    def test_custom_timeout_configuration(self, mock_client_cls):
        get_client()
        mock_client_cls.assert_called_once()
        _, kwargs = mock_client_cls.call_args
        self.assertEqual(kwargs["http_options"].timeout, 45000)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_test_key", "GEMINI_TIMEOUT_SECONDS": "not_a_number"}, clear=True)
    @patch("app.genai.Client")
    def test_fallback_timeout_on_invalid_env(self, mock_client_cls):
        get_client()
        mock_client_cls.assert_called_once()
        _, kwargs = mock_client_cls.call_args
        self.assertEqual(kwargs["http_options"].timeout, 30000)


class TestApiErrorHandling(unittest.TestCase):
    """Tests for categorizing and formatting external API errors."""

    def test_timeout_504_formatting(self):
        exc = Exception("504 DEADLINE_EXCEEDED: Deadline expired")
        msg = format_api_error(exc, "gemini-3.8-flash")
        self.assertIn("504", msg)
        self.assertIn("gemini-3.8-flash", msg)
        self.assertIn("gemini-flash-lite-latest", msg)

    def test_rate_limit_429_formatting(self):
        exc = Exception("429 RESOURCE_EXHAUSTED: Quota exceeded")
        msg = format_api_error(exc, "gemini-3.5-flash-lite")
        self.assertIn("429", msg)
        self.assertIn("Rate Limit", msg)

    def test_auth_failure_formatting(self):
        exc = Exception("401 UNAUTHENTICATED: API_KEY_INVALID")
        msg = format_api_error(exc, "gemini-flash-lite-latest")
        self.assertIn("Authentication Failure", msg)
        self.assertIn("GEMINI_API_KEY", msg)

    def test_model_404_formatting(self):
        exc = Exception("404 NOT_FOUND: model models/gemini-2.5-flash is no longer available")
        msg = format_api_error(exc, "gemini-2.5-flash")
        self.assertIn("404", msg)
        self.assertIn("gemini-2.5-flash", msg)

    def test_network_connection_error_formatting(self):
        exc = Exception("ConnectionError: Failed to establish a new connection")
        msg = format_api_error(exc, "gemini-flash-lite-latest")
        self.assertIn("Network Connection Error", msg)


class TestAgentExecutionMock(unittest.TestCase):
    """Tests for agent chat session and tool invocation workflow (mocked)."""

    def test_ask_agent_successful_call(self):
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "400"
        mock_chat.send_message.return_value = mock_response
        mock_client.chats.create.return_value = mock_chat

        result = ask_agent(mock_client, "What is 25 * 16?", "gemini-flash-lite-latest")
        self.assertEqual(result, "400")
        mock_client.chats.create.assert_called_once()
        mock_chat.send_message.assert_called_once_with("What is 25 * 16?")

    def test_ask_agent_empty_response_fallback(self):
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = ""
        mock_chat.send_message.return_value = mock_response
        mock_client.chats.create.return_value = mock_chat

        result = ask_agent(mock_client, "Hello", "gemini-flash-lite-latest")
        self.assertIn("did not generate a text summary", result)


if __name__ == "__main__":
    unittest.main()

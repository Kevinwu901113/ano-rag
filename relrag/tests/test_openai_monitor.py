import unittest
from unittest.mock import MagicMock, patch
import logging
import sys
from loguru import logger as loguru_logger
import relrag

# Configure logging to capture warnings
logging.basicConfig(level=logging.INFO)

from relrag.config.config_loader import ConfigLoader
from relrag.utils.openai_answer import generate_openai_answer
from relrag.utils.context_budget import budget_answer_prompt

class TestOpenAIConfigAndMonitor(unittest.TestCase):
    def test_config_loading(self):
        loader = ConfigLoader()
        cfg = loader.load_config()
        # print(f"DEBUG: config answer section: {cfg.get('answer')}")
        openai_cfg = cfg.get("openai", {})
        self.assertEqual(openai_cfg.get("max_tokens"), 8192, "max_tokens should be 8192")
        self.assertEqual(openai_cfg.get("temperature"), 0.7, "temperature should be 0.7")
        self.assertEqual(cfg.get("answer", {}).get("max_evidence_items"), 12)
        self.assertEqual(cfg.get("answer", {}).get("max_evidence_tokens"), 512)

    @patch("relrag.utils.openai_answer.chat_completion")
    @patch("relrag.utils.openai_answer.log_budget_event")
    @patch("relrag.utils.openai_answer.get_active_llm_stats")
    def test_generate_answer_monitoring(self, mock_stats, mock_log, mock_chat):
        mock_chat.return_value = "Test Answer"
        mock_stats.return_value = MagicMock()
        
        # Capture loguru logs
        logs = []
        sink_id = loguru_logger.add(lambda msg: logs.append(msg))
        
        try:
            # Mock config with small tokens
            openai_cfg = {
                "model": "gpt-4",
                "api_key": "fake",
                "max_tokens": 100, # Small tokens to trigger warning
                "temperature": 0.7
            }
            
            generate_openai_answer("Question?", [], openai_cfg)
            self.assertTrue(any("Requested max_tokens 100 is very small" in str(o) for o in logs))
            
            # Verify call args - max_tokens should be adjusted to 256
            args, kwargs = mock_chat.call_args
            self.assertGreaterEqual(kwargs.get("max_tokens", 0), 256)
        finally:
            loguru_logger.remove(sink_id)

    @patch("relrag.utils.openai_answer.chat_completion")
    @patch("relrag.utils.openai_answer.log_budget_event")
    @patch("relrag.utils.openai_answer.get_active_llm_stats")
    def test_empty_response_warning(self, mock_stats, mock_log, mock_chat):
        mock_chat.return_value = ""
        mock_stats.return_value = MagicMock()
        openai_cfg = {"model": "gpt-4", "api_key": "fake", "max_tokens": 1024}
        
        logs = []
        sink_id = loguru_logger.add(lambda msg: logs.append(msg))
        
        try:
            generate_openai_answer("Question?", [], openai_cfg)
            self.assertTrue(any("OpenAI returned empty response" in str(o) for o in logs))
        finally:
            loguru_logger.remove(sink_id)

    def test_context_budgeting(self):
        # Test if budgeting respects new limits
        report = budget_answer_prompt(
            "Q", 
            [{"canonical": "A", "evidence": "A"*1000}], 
            prompt_name="answerer_openai.txt", 
            label_instruction="", 
            system_prompt="",
            requested_max_tokens=8192
        ).report
        
        # Check if report fields exist
        self.assertIsNotNone(report.effective_max_tokens)
        self.assertIsNotNone(report.estimated_input_tokens)

if __name__ == "__main__":
    unittest.main()

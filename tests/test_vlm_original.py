import unittest
from unittest.mock import Mock
from vlm_language import judge_original


class OriginalJudgmentTest(unittest.TestCase):
    def test_yes_no_without_reason_are_valid(self):
        for response, verdict in [('yes', 'confirmed'), ('NO.', 'false_alarm')]:
            generate = Mock(return_value=response)
            result = judge_original(generate, 'original prompt')
            self.assertEqual(result[0], verdict)
            self.assertEqual(result[1], response)
            generate.assert_called_once_with('original prompt', True)

    def test_missing_or_ambiguous_answer_is_not_false_alarm(self):
        for response in ['', 'yes or no', 'uncertain', 'no_answer_generated']:
            self.assertEqual(judge_original(Mock(return_value=response), 'prompt')[0], 'error')

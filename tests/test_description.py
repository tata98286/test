import unittest
from unittest.mock import Mock
from independent_description import judge_with_description, SCENE_PROMPT


class DescriptionTest(unittest.TestCase):
    def test_no_independent_absence(self):
        gen = Mock(side_effect=['no', '{"description":"A street lamp illuminates the empty road without flames or smoke.","visible_combustion":"absent"}'])
        verdict, text, info = judge_with_description(gen, 'original question')
        self.assertEqual(verdict, 'false_alarm')
        self.assertEqual(info['consistency'], 'match')
        self.assertIn('street lamp', text)
        self.assertEqual(gen.call_args_list[1].args, (SCENE_PROMPT, True))
        self.assertNotIn('original question', gen.call_args_list[1].args[0])

    def test_conflicting_description_does_not_overwrite_verdict(self):
        gen = Mock(side_effect=['no', '{"description":"Orange flames and dark smoke surround the vehicle.","visible_combustion":"present"}'])
        verdict, text, info = judge_with_description(gen, 'original')
        self.assertEqual(verdict, 'false_alarm')
        self.assertEqual(info['consistency'], 'mismatch')
        self.assertIn('mismatch', text)

    def test_description_exception_preserves_yes(self):
        verdict, text, info = judge_with_description(Mock(side_effect=['yes', RuntimeError('failure')]), 'original')
        self.assertEqual(verdict, 'confirmed')
        self.assertEqual(info['description_status'], 'failed')
        self.assertIn('Description unavailable', text)

    def test_malformed_description_keeps_raw_audit(self):
        verdict, _, info = judge_with_description(Mock(side_effect=['yes', 'YES']), 'original')
        self.assertEqual(verdict, 'confirmed')
        self.assertEqual(info['attempts'][1]['response'], 'YES')
        self.assertEqual(info['description_status'], 'failed')

    def test_unclear_description_is_not_mismatch(self):
        gen = Mock(side_effect=['yes', '{"description":"A dark obstruction hides most of the vehicle.","visible_combustion":"unclear"}'])
        verdict, _, info = judge_with_description(gen, 'original')
        self.assertEqual(verdict, 'confirmed')
        self.assertEqual(info['consistency'], 'unresolved')

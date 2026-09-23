import unittest
from vlm_language import judge_and_translate, parse_english

class LanguageTest(unittest.TestCase):
    def test_english_then_text_only_translation(self):
        calls=[]
        def generate(prompt, images):
            calls.append(images)
            return 'YES\nOrange flames are visible beside the vehicle.' if images else '차량 옆에 주황색 불꽃이 보입니다.'
        verdict, answer, details=judge_and_translate(generate,'Review images')
        self.assertEqual(verdict,'confirmed')
        self.assertEqual(calls,[True,False])
        self.assertEqual(details['translation_status'],'complete')
        self.assertIn('Orange',answer)

    def test_translation_failure_preserves_no(self):
        def generate(prompt, images):
            if images: return 'NO\nThe bright patch is a lamp reflecting on the window.'
            raise RuntimeError('translation failed')
        verdict, _, details=judge_and_translate(generate,'Review images')
        self.assertEqual(verdict,'false_alarm')
        self.assertEqual(details['translation_status'],'failed')

    def test_rejudge_both_fields_instead_of_locking_first_verdict(self):
        outputs=iter(['NO\nOne short English sentence describing evidence.', 'YES\nOrange flames are visible above the car hood.', '차량 보닛 위로 주황색 불꽃이 보입니다.'])
        verdict, _, details=judge_and_translate(lambda p,i:next(outputs),'Review images')
        self.assertEqual(verdict,'confirmed')
        self.assertEqual(len(details['attempts']),3)

    def test_echo_is_error_not_uncertain(self):
        result=judge_and_translate(lambda p,i:'YES, NO, or UNCERTAIN','Review images')
        self.assertEqual(result[0],'error')
        self.assertIsNone(parse_english('UNCERTAIN\nI cannot be completely sure of the answer.'))

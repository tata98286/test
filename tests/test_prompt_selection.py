import unittest
from unittest.mock import patch
import app
from flask import render_template
class PromptSelectionTest(unittest.TestCase):
    def test_fixed_ignores_custom_text(self):
        mode,prompt,version=app.select_vlm_prompt({'vlm_test_prompt':'custom'})
        self.assertEqual((mode,prompt,version),('fixed',app.VLM_PROMPT,app.VLM_PROMPT_VERSION))
    def test_custom_snapshot_has_own_version(self):
        mode,prompt,version=app.select_vlm_prompt({'vlm_prompt_mode':'test','vlm_test_prompt':'  Test fire. Answer yes or no.  '})
        self.assertEqual(mode,'test');self.assertEqual(prompt,'Test fire. Answer yes or no.')
        self.assertNotEqual(version,app.VLM_PROMPT_VERSION)
    def test_invalid_selection_rejected(self):
        for form in [{'vlm_prompt_mode':'unknown'},{'vlm_prompt_mode':'test'},{'vlm_prompt_mode':'test','vlm_test_prompt':'x'*4001}]:
            with self.assertRaises(ValueError):app.select_vlm_prompt(form)
    def test_worker_uses_snapshot_and_records_it(self):
        job={'vlm_prompt_mode':'test','vlm_prompt':'Test prompt','vlm_prompt_version':'version-test','vlm_events':{123:{'status':'queued'}},'pending_vlm':1}
        with patch.object(app,'classify_with_vlm',return_value=('confirmed','yes')) as classify, patch.object(app,'open_db_connection') as db:
            app.run_vlm_for_event(job,123,[])
            self.assertEqual(classify.call_args.kwargs['prompt'],'Test prompt')
            calls=db.return_value.cursor.return_value.__enter__.return_value.execute.call_args_list
            self.assertEqual(calls[0].args[1],('confirmed','yes','version-test','Test prompt',123))
        self.assertEqual(job['pending_vlm'],0)
    def test_modal_escapes_fixed_prompt(self):
        with app.app.test_request_context('/'):
            app.session['csrf_token']='test'
            html=render_template('home.html',user={'username':'test','role':'USER'},report=None,job_token=None,events=[],current_job={},fixed_vlm_prompt='</textarea><script>alert(1)</script>',fixed_vlm_version='test')
            self.assertNotIn('</textarea><script>alert(1)</script>',html)
            self.assertIn('vlm-prompt-dialog',html)
            self.assertIn('name="vlm_prompt_mode" value="fixed"',html)
if __name__=='__main__':unittest.main()

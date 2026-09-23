"""Standalone scene description: never includes or changes the first verdict."""
import json
import re
from vlm_language import judge_original

SCENE_PROMPT = '''Describe only what is visible in this image in one short English sentence.
Mention the objects, location, and appearance of any flames or smoke, if visible.
Do not invent details or define fire. Do not assume an automated alert is correct.
Return a JSON object with these two fields:
"description": your concrete visual observation in English,
"visible_combustion": "present", "absent", or "unclear".
Use present for visible flames or combustion smoke, absent when neither is visible,
and unclear when the image does not let you distinguish them.'''


def parse_description(raw):
    text = raw.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.I)
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict):
        return None
    observation = value.get('description')
    evidence = value.get('visible_combustion')
    if not isinstance(observation, str) or len(observation.split()) < 5:
        return None
    if evidence not in {'present', 'absent', 'unclear'}:
        return None
    if not re.search('[A-Za-z]', observation) or re.search('[가-힣]', observation):
        return None
    if any(s in observation.lower() for s in ('your concrete', 'describe only', 'is defined as')):
        return None
    return observation.strip(), evidence


def judge_with_description(generate, prompt):
    verdict, original, details = judge_original(generate, prompt)
    scene = None
    try:
        # Fresh image request; neither the original prompt nor answer is passed here.
        raw = generate(SCENE_PROMPT, True)
        details['attempts'].append({'stage': 'independent_description', 'prompt': SCENE_PROMPT, 'response': raw})
        scene = parse_description(raw)
    except Exception as exc:
        details['attempts'].append({'stage': 'independent_description', 'prompt': SCENE_PROMPT, 'error': type(exc).__name__})
    consistency = 'not_checked'
    description = None
    if scene:
        description, evidence = scene
        expected = {'confirmed': 'present', 'false_alarm': 'absent'}.get(verdict)
        if expected and evidence != 'unclear':
            consistency = 'match' if expected == evidence else 'mismatch'
        else:
            consistency = 'unresolved'
    details.update(description=description, consistency=consistency, description_status='complete' if scene else 'failed')
    details['attempts'].append({'stage': 'consistency', 'status': consistency,
                                'description': description, 'original_verdict': verdict})
    displayed = original + '\nScene observation: ' + (description or 'Description unavailable; original verdict retained.')
    if consistency == 'mismatch':
        displayed += '\n[Judgment/description mismatch — review required]'
    elif consistency == 'unresolved':
        displayed += '\n[Description comparison inconclusive]'
    return verdict, displayed, details

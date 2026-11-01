"""Check the homework's data and grading without downloading a model or using a GPU."""
import ast
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
notebook = json.loads((ROOT / 'notebooks/lec02_lab_faq.ipynb').read_text(encoding='utf-8'))
code = {i: ''.join(c['source']) for i, c in enumerate(notebook['cells']) if c['cell_type'] == 'code'}
for i, source in code.items():
    if not any(line.startswith(('!', '%')) for line in source.splitlines()):
        ast.parse(source, filename=f'cell_{i}')

with tempfile.TemporaryDirectory() as tmp:
    env = {'json': json, 'WORK': Path(tmp)}
    exec(compile(code[7], '<data>', 'exec'), env)
    train, valid, test, gold = (env[k] for k in ('train', 'valid', 'test', 'gold'))
    assert [len(s) for s in (train, valid, test)] == [14, 7, 21]
    assert [sum(e['needs_review'] for e in s) for s in (train, valid, test)] == [4, 2, 6]
    questions = [{e['question'] for e in s} for s in (train, valid, test)]
    assert all(not a & b for i, a in enumerate(questions) for b in questions[i + 1:])
    assert set(gold) == questions[2]
    for example in train + valid + test:
        target = json.loads(example['target'])
        assert set(target) == {'answer', 'needs_review'}
        assert isinstance(target['answer'], str)
        assert type(target['needs_review']) is bool
        assert target['needs_review'] == example['needs_review']
        assert example['faq'] == env['FAQ_TEXT']

    functions = [n for n in ast.parse(code[10]).body if isinstance(n, ast.FunctionDef)]
    grading = ast.Module(body=[n for n in functions if n.name in ('check_format', 'check_content')], type_ignores=[])
    exec(compile(grading, '<grading>', 'exec'), env)
    fmt, content = env['check_format'], env['check_content']
    for example in test:
        ok, _, obj = fmt(example['target'])
        assert ok and content(obj, gold[example['question']])
    assert not fmt('回答です')[0]
    assert not fmt('[]')[0]
    assert not fmt('{"answer":"確認が必要"}')[0]
    assert not fmt('{"answer":"確認が必要","needs_review":"true"}')[0]
    assert not fmt('{"answer":"確認が必要","needs_review":true,"extra":1}')[0]
    assert not content({'answer': 123, 'needs_review': False}, {'needs_review': False, 'must_include': ['3日']})
    assert not content({'answer': '3日', 'needs_review': 0}, {'needs_review': False, 'must_include': ['3日']})
    assert not content({'answer': '確認が必要', 'needs_review': False}, {'needs_review': True, 'must_include': ['確認']})
    assert not content({'answer': '5日', 'needs_review': False}, {'needs_review': False, 'must_include': ['3日']})
    assert '"FAQにある": has_evidence' in code[17]
    assert 'with tuned.disable_adapter()' in code[17]
    assert 'training_stats' in code[19] and 'fictional_lab_equipment_faq_v1' in code[19]
    assert 'drive.mount' not in code[21]

print('PASS: notebook syntax, 14/7/21 split, no question overlap, gold coverage, grading edge cases, result metadata')

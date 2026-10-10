"""Execute the workflow's SARIF validation against valid and broken outputs."""
import json
from pathlib import Path
import subprocess

import pytest
import yaml


WORKFLOWS = Path(__file__).parents[1] / '.github/workflows'


def report(tool):
    return {'version': '2.1.0', 'runs': [{'tool': {'driver': {'name': tool}}}]}


@pytest.mark.parametrize('workflow', ['security-scan.yml', 'defender-for-devops.yml'])
@pytest.mark.parametrize('failure', [None, 'missing', 'console', 'empty_runs', 'wrong_tool'])
def test_workflow_requires_both_real_sarif_reports(tmp_path, failure, workflow):
    steps = yaml.safe_load((WORKFLOWS / workflow).read_text())['jobs']['security-scan']['steps']
    validate = next(step for step in steps if step.get('id') == 'verify_sarif')['run']
    (tmp_path / 'bandit.sarif').write_text(json.dumps(report('Bandit')))
    (tmp_path / 'checkov.sarif').write_text(json.dumps(report('Checkov')))
    if failure == 'missing':
        (tmp_path / 'bandit.sarif').unlink()
        (tmp_path / 'unrelated.sarif').write_text(json.dumps(report('Other')))
    elif failure == 'console':
        (tmp_path / 'checkov.sarif').write_text('Checkov console banner\nWrote results.sarif')
    elif failure == 'empty_runs':
        (tmp_path / 'checkov.sarif').write_text(json.dumps({'version': '2.1.0', 'runs': []}))
    elif failure == 'wrong_tool':
        (tmp_path / 'bandit.sarif').write_text(json.dumps(report('Checkov')))
    result = subprocess.run(['bash', '-c', validate], cwd=tmp_path, capture_output=True, text=True)
    assert (result.returncode == 0) == (failure is None), result.stderr
    for step in steps:
        if step.get('uses', '').startswith('github/codeql-action/upload-sarif@'):
            assert "steps.verify_sarif.outcome == 'success'" in step['if']

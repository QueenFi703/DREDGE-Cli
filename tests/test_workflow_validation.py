"""Keep workflow parsing and deployment/reporting boundaries explicit."""
from pathlib import Path
import subprocess

import yaml


WORKFLOWS = Path(__file__).parents[1] / '.github/workflows'


def test_all_workflows_parse():
    for path in WORKFLOWS.glob('*.yml'):
        value = yaml.safe_load(path.read_text())
        assert isinstance(value.get('jobs'), dict), path


def test_pages_only_deploys_on_explicit_dispatch(tmp_path):
    workflow = yaml.safe_load((WORKFLOWS / 'deploy.yml').read_text())
    # PyYAML's YAML 1.1 parser recognizes the Actions key `on` as True.
    assert 'workflow_dispatch' in workflow[True]
    assert workflow['permissions'] == {'contents': 'read'}
    assert workflow['jobs']['deploy']['if'] == "github.event_name == 'workflow_dispatch'"
    assert workflow['jobs']['deploy']['needs'] == 'build'
    build = workflow['jobs']['build']
    assert all('deploy-pages' not in step.get('uses', '') for step in build['steps'])
    script = next(step['run'] for step in build['steps']
                  if step['name'] == 'Create API proxy documentation')
    result = subprocess.run(['bash', '-c', script], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'docs/API_ENDPOINTS.md').read_text().startswith('# DREDGE API Endpoints')


def test_gate_does_not_activate_outbound_reporting():
    workflow = yaml.safe_load((WORKFLOWS / 'dredge-gate-agent.yml').read_text())
    assert workflow['permissions'] == {'contents': 'read', 'actions': 'read'}
    steps = {step['name']: step for step in workflow['jobs']['gate-agent']['steps']}
    assert steps['Annotate Pull Request']['if'] == '${{ false }}'
    assert steps['Export Telemetry']['if'] == '${{ false }}'
    classify = steps['Intent Classification']
    assert classify['env']['CHANGED_FILES'] == '${{ steps.changes.outputs.changed_files }}'
    assert '${{ steps.changes.outputs.changed_files }}' not in classify['run']

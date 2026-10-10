"""The Railway check only builds; it cannot publish or deploy the image."""
from pathlib import Path

import yaml


def test_railway_check_is_build_only():
    path = Path(__file__).parents[1] / '.github/workflows/railway-build.yml'
    workflow = yaml.safe_load(path.read_text())
    triggers = workflow.get('on', workflow.get(True))
    assert set(triggers) == {'pull_request', 'workflow_dispatch'}
    assert triggers['pull_request']['branches'] == ['main']
    assert workflow['permissions'] == {'contents': 'read'}
    steps = workflow['jobs']['railway-build']['steps']
    assert len(steps) == 2
    assert steps[0]['uses'] == 'actions/checkout@v6'
    assert steps[1]['run'] == 'docker build --file Dockerfile.railway --tag dredge-studio:railway-ci .'
    assert 'secrets.' not in path.read_text()

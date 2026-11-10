"""Keep container build targets and CI smoke tests aligned with the runtime."""
from pathlib import Path
import re

import pytest
import yaml


ROOT = Path(__file__).parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text()
STAGES = dict(
    (name, base)
    for base, name in re.findall(r"^FROM (\S+) AS (\S+)$", DOCKERFILE, re.MULTILINE)
)


def workflow(filename):
    return yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())


@pytest.mark.parametrize("filename,job", [
    ("docker-test.yml", "validate-docker"),
    ("docker-publish.yml", "build-and-push"),
])
def test_ci_build_targets_exist(filename, job):
    matrix = workflow(filename)["jobs"][job]["strategy"]["matrix"]
    targets = {item["target"] for item in matrix["include"]}
    assert targets == {"cpu-build", "gpu-build", "dev"}
    assert targets <= STAGES.keys()


def test_gpu_uses_the_matching_python_runtime_and_cuda_wheel():
    assert STAGES["gpu"] == "runtime"
    assert STAGES["cpu-build"] == "runtime"
    assert STAGES["gpu-build"] == "gpu"
    gpu = DOCKERFILE.split("FROM runtime AS gpu\n", 1)[1]
    assert "https://download.pytorch.org/whl/cu128" in gpu
    assert 'torch.version.cuda == "12.8"' in gpu
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility" in gpu
    assert "NVIDIA_REQUIRE_CUDA=\"cuda>=12.8\"" in gpu


def test_cpu_installs_and_verifies_cpu_torch_before_other_requirements():
    builder = DOCKERFILE.split("FROM python:3.14-slim AS runtime", 1)[0]
    assert "https://download.pytorch.org/whl/cpu" in builder
    assert builder.index("https://download.pytorch.org/whl/cpu") < builder.index("pip install -r requirements.txt")
    assert "torch.version.cuda is None" in builder


def test_smoke_tests_map_to_the_actual_server_port():
    assert '"--port", "8001"' in DOCKERFILE
    validation = workflow("docker-test.yml")["jobs"]["validate-docker"]["steps"]
    smoke_runs = [step["run"] for step in validation if step.get("name", "").startswith("Test ")]
    assert len(smoke_runs) == 2
    for command in smoke_runs:
        host, container = re.search(r"-p (\d+):(\d+)", command).groups()
        assert container == "8001"
        assert f"http://localhost:{host}/health" in command
    matrix = workflow("docker-publish.yml")["jobs"]["test-images"]["strategy"]["matrix"]["include"]
    for image in matrix:
        host, container = image["port_map"].split(":")
        assert container == "8001"
        assert image["health_url"] == f"http://localhost:{host}/health"

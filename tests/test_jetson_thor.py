"""
Tests for the Jetson Thor hardware adapter, AI event bus,
autonomous agents, MCP agent server, and run-agent CLI command.
"""
import json
import sys

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Hardware adapter – JetsonThor
# ---------------------------------------------------------------------------


class TestJetsonThor:
    """Tests for dredge.hardware.jetson_thor.JetsonThor."""

    def test_import(self):
        from dredge.hardware.jetson_thor import JetsonThor  # noqa: F401

    def test_default_device_is_cpu_or_cuda(self):
        from dredge.hardware.jetson_thor import JetsonThor

        thor = JetsonThor()
        assert thor.device in ("cpu", "cuda")

    def test_auto_device_resolves(self):
        from dredge.hardware.jetson_thor import JetsonThor

        thor = JetsonThor(device="auto")
        assert thor.device in ("cpu", "cuda")

    def test_explicit_cpu_device(self):
        from dredge.hardware.jetson_thor import JetsonThor

        thor = JetsonThor(device="cpu")
        assert thor.device == "cpu"

    def test_cuda_request_without_gpu_falls_back_to_cpu(self):
        from dredge.hardware.jetson_thor import JetsonThor

        if not torch.cuda.is_available():
            thor = JetsonThor(device="cuda")
            assert thor.device == "cpu"
        else:
            pytest.skip("CUDA is available; fallback path not exercised")

    def test_compute_returns_tensor(self):
        from dredge.hardware.jetson_thor import JetsonThor

        thor = JetsonThor(device="cpu")
        model = nn.Linear(4, 2)
        data = torch.randn(1, 4)
        result = thor.compute(model, data)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 2)

    def test_compute_moves_data_to_device(self):
        from dredge.hardware.jetson_thor import JetsonThor

        thor = JetsonThor(device="cpu")
        model = nn.Linear(4, 2)
        data = torch.randn(1, 4)
        result = thor.compute(model, data)
        assert result.device.type == "cpu"

    def test_device_info_returns_dict(self):
        from dredge.hardware.jetson_thor import JetsonThor

        thor = JetsonThor(device="cpu")
        info = thor.device_info()
        assert isinstance(info, dict)
        assert "device" in info
        assert "cuda_available" in info
        assert info["device"] == "cpu"

    def test_hardware_package_import(self):
        from dredge.hardware import JetsonThor  # noqa: F401


# ---------------------------------------------------------------------------
# Events – AIEvent + dispatch
# ---------------------------------------------------------------------------


class TestAIEvent:
    """Tests for dredge.events.ai_event."""

    def test_ai_event_creation(self):
        from dredge.events.ai_event import AIEvent

        model = nn.Linear(4, 2)
        data = torch.randn(1, 4)
        event = AIEvent(model=model, input_data=data)
        assert event.model is model
        assert event.input_data is data

    def test_dispatch_ai_event_returns_tensor(self):
        from dredge.events.ai_event import AIEvent, dispatch

        model = nn.Linear(4, 2)
        data = torch.randn(1, 4)
        event = AIEvent(model=model, input_data=data)
        result = dispatch(event)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 2)

    def test_dispatch_unknown_event_raises_type_error(self):
        from dredge.events.ai_event import dispatch

        with pytest.raises(TypeError, match="unknown event type"):
            dispatch("not-an-event")

    def test_events_package_import(self):
        from dredge.events import AIEvent, dispatch  # noqa: F401


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class TestVisionAgent:
    """Tests for dredge.agents.vision_agent.VisionAgent."""

    def test_default_construction(self):
        from dredge.agents.vision_agent import VisionAgent

        agent = VisionAgent()
        assert agent.model is not None

    def test_run_returns_tensor(self):
        from dredge.agents.vision_agent import VisionAgent

        agent = VisionAgent(input_dim=8, output_dim=4)
        data = torch.randn(1, 8)
        result = agent.run(data)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 4)

    def test_describe(self):
        from dredge.agents.vision_agent import VisionAgent

        agent = VisionAgent()
        desc = agent.describe()
        assert desc["agent"] == "VisionAgent"
        assert "role" in desc

    def test_custom_model(self):
        from dredge.agents.vision_agent import VisionAgent

        custom = nn.Linear(16, 8)
        agent = VisionAgent(model=custom)
        data = torch.randn(1, 16)
        result = agent.run(data)
        assert result.shape == (1, 8)


class TestPlannerAgent:
    """Tests for dredge.agents.planner_agent.PlannerAgent."""

    def test_default_construction(self):
        from dredge.agents.planner_agent import PlannerAgent

        agent = PlannerAgent()
        assert agent.model is not None

    def test_run_returns_tensor(self):
        from dredge.agents.planner_agent import PlannerAgent

        agent = PlannerAgent(input_dim=16, num_actions=4)
        data = torch.randn(1, 16)
        result = agent.run(data)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 4)

    def test_describe(self):
        from dredge.agents.planner_agent import PlannerAgent

        agent = PlannerAgent()
        desc = agent.describe()
        assert desc["agent"] == "PlannerAgent"


class TestReasoningAgent:
    """Tests for dredge.agents.reasoning_agent.ReasoningAgent."""

    def test_default_construction(self):
        from dredge.agents.reasoning_agent import ReasoningAgent

        agent = ReasoningAgent()
        assert agent.model is not None

    def test_run_returns_scalar_tensor(self):
        from dredge.agents.reasoning_agent import ReasoningAgent

        agent = ReasoningAgent(input_dim=4, output_dim=1)
        data = torch.randn(1, 4)
        result = agent.run(data)
        assert isinstance(result, torch.Tensor)
        # Sigmoid output should be in [0, 1]
        assert 0.0 <= result.item() <= 1.0

    def test_describe(self):
        from dredge.agents.reasoning_agent import ReasoningAgent

        agent = ReasoningAgent()
        desc = agent.describe()
        assert desc["agent"] == "ReasoningAgent"


class TestAgentsPackage:
    """Tests for dredge.agents package-level imports."""

    def test_package_exports(self):
        from dredge.agents import PlannerAgent, ReasoningAgent, VisionAgent  # noqa: F401


class TestFullPipeline:
    """End-to-end test of the Vision → Planner → Reasoning chain."""

    def test_pipeline_produces_decision(self):
        from dredge.agents.planner_agent import PlannerAgent
        from dredge.agents.reasoning_agent import ReasoningAgent
        from dredge.agents.vision_agent import VisionAgent

        sensor = torch.randn(1, 64)

        vision = VisionAgent(input_dim=64, output_dim=32)
        planner = PlannerAgent(input_dim=32, num_actions=8)
        reasoning = ReasoningAgent(input_dim=8, output_dim=1)

        embedding = vision.run(sensor)
        action_logits = planner.run(embedding)
        decision = reasoning.run(action_logits)

        assert embedding.shape == (1, 32)
        assert action_logits.shape == (1, 8)
        assert decision.shape == (1, 1)
        assert 0.0 <= decision.item() <= 1.0


# ---------------------------------------------------------------------------
# MCP Agent Server (FastAPI)
# ---------------------------------------------------------------------------


class TestMCPAgentServer:
    """Tests for dredge.mcp.server."""

    def test_create_app(self):
        from dredge.mcp.server import create_mcp_agent_app

        app = create_mcp_agent_app(device="cpu")
        assert app is not None

    def test_health_endpoint(self):
        from fastapi.testclient import TestClient

        from dredge.mcp.server import create_mcp_agent_app

        client = TestClient(create_mcp_agent_app(device="cpu"))
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert "device_info" in body
        assert "agents" in body

    def test_compute_endpoint(self):
        from fastapi.testclient import TestClient

        from dredge.mcp.server import create_mcp_agent_app

        client = TestClient(create_mcp_agent_app(device="cpu"))
        payload = {"data": [0.1] * 64, "input_dim": 64, "output_dim": 32}
        resp = client.post("/compute", json=payload)
        assert resp.status_code == 200
        assert "result" in resp.json()

    def test_agent_vision_endpoint(self):
        from fastapi.testclient import TestClient

        from dredge.mcp.server import create_mcp_agent_app

        client = TestClient(create_mcp_agent_app(device="cpu"))
        payload = {"data": [0.5] * 64}
        resp = client.post("/agent/vision", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["agent"] == "VisionAgent"
        assert "result" in body

    def test_agent_planner_endpoint(self):
        from fastapi.testclient import TestClient

        from dredge.mcp.server import create_mcp_agent_app

        client = TestClient(create_mcp_agent_app(device="cpu"))
        payload = {"data": [0.5] * 32}
        resp = client.post("/agent/planner", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["agent"] == "PlannerAgent"

    def test_agent_reasoning_endpoint(self):
        from fastapi.testclient import TestClient

        from dredge.mcp.server import create_mcp_agent_app

        client = TestClient(create_mcp_agent_app(device="cpu"))
        payload = {"data": [0.5] * 8}
        resp = client.post("/agent/reasoning", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["agent"] == "ReasoningAgent"

    def test_pipeline_endpoint(self):
        from fastapi.testclient import TestClient

        from dredge.mcp.server import create_mcp_agent_app

        client = TestClient(create_mcp_agent_app(device="cpu"))
        payload = {
            "data": [0.1] * 64,
            "vision_output_dim": 32,
            "planner_num_actions": 8,
        }
        resp = client.post("/pipeline", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "embedding" in body
        assert "action_logits" in body
        assert "decision" in body
        assert body["pipeline"] == "Vision → Planner → Reasoning"


# ---------------------------------------------------------------------------
# CLI – run-agent command
# ---------------------------------------------------------------------------


class TestRunAgentCLI:
    """Tests for the dredge-cli run-agent command."""

    def _run(self, *args):
        from dredge.cli import main

        return main(list(args))

    def test_run_agent_pipeline_default(self):
        rc = self._run("run-agent", "--device", "cpu")
        assert rc == 0

    def test_run_agent_vision(self):
        rc = self._run("run-agent", "--agent", "vision", "--device", "cpu", "--input-dim", "8")
        assert rc == 0

    def test_run_agent_planner(self):
        rc = self._run("run-agent", "--agent", "planner", "--device", "cpu", "--input-dim", "8")
        assert rc == 0

    def test_run_agent_reasoning(self):
        rc = self._run("run-agent", "--agent", "reasoning", "--device", "cpu", "--input-dim", "8")
        assert rc == 0

    def test_run_agent_with_input_values(self):
        rc = self._run(
            "run-agent",
            "--agent", "pipeline",
            "--input", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8",
            "--device", "cpu",
        )
        assert rc == 0

    def test_run_agent_json_output(self, capsys):
        rc = self._run("run-agent", "--device", "cpu", "--json")
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "pipeline" in data or "agent" in data

    def test_run_agent_invalid_input(self, capsys):
        rc = self._run("run-agent", "--input", "not,valid,floats", "--device", "cpu")
        assert rc == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_run_agent_help(self, capsys):
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "dredge", "run-agent", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--agent" in result.stdout

    def test_agent_server_help(self, capsys):
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "dredge", "agent-server", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--host" in result.stdout
        assert "--port" in result.stdout

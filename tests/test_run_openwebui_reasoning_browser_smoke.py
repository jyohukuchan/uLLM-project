from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/run-openwebui-reasoning-browser-smoke.py"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reasoning_browser_runner", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()

SESSION_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjQwMDAwMDAwMDB9.signature"


def test_v2_dispatch_binds_reasoning_browser_run_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.json"
    observed: dict[str, object] = {}

    def construct(**arguments: object) -> object:
        observed.update(arguments)
        return type(
            "Binding",
            (),
            {"candidate": type("Candidate", (), {"path": candidate})()},
        )()

    monkeypatch.setattr(TOOL, "optional_v2_binding", construct)
    manifest, binding = TOOL._select_manifest_and_binding(
        active_binding_mode="v2",
        manifest=None,
        candidate_served_model_manifest=candidate,
        active_served_model_manifest=tmp_path / "active.json",
        expected_served_model_manifest_sha256="a" * 64,
        campaign_authorization=tmp_path / "authorization.json",
        run_id="reasoning-browser-run",
        output=tmp_path / "browser-output.json",
    )

    assert binding is not None
    assert manifest == candidate
    assert observed["campaign_name"] == "reasoning_browser"
    assert observed["run_id"] == "reasoning-browser-run"
    assert observed["final_path"] == tmp_path / "browser-output.json"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def evidence(model_id: str, switch_model_id: str | None = None) -> dict:
    primary = digest(model_id)

    def request(model_hash: str, suffix: str) -> dict:
        return {
            "sha256": digest(f"request-{suffix}"),
            "utf8_bytes": 128,
            "model_id_sha256": model_hash,
            "has_reasoning_content_key": True,
            "assistant_has_reasoning_content": False,
        }

    result = {
        "schema_version": "ullm.openwebui.reasoning_browser_smoke.v2",
        "model_id_sha256": primary,
        "first_answer": {"utf8_bytes": 20, "sha256": "c" * 64},
        "expanded_view": {"utf8_bytes": 40, "sha256": "f" * 64},
        "second_answer": {"utf8_bytes": 21, "sha256": "d" * 64},
        "reasoning_details_expanded": True,
        "provider_request_count": 2,
        "provider_requests": [
            request(primary, "one"),
            request(primary, "two"),
        ],
        "hidden_reasoning_reinserted": False,
        "page_error_count": 0,
        "page_error_digests": [],
    }
    if switch_model_id is not None:
        switched = digest(switch_model_id)
        result.update(
            {
                "provider_switch_performed": True,
                "provider_switch_model_id_sha256": switched,
                "provider_switch_answer": {"utf8_bytes": 22, "sha256": "3" * 64},
                "provider_return_performed": True,
                "provider_return_model_id_sha256": primary,
                "provider_return_answer": {"utf8_bytes": 23, "sha256": "6" * 64},
                "provider_request_count": 4,
                "provider_requests": [
                    request(primary, "one"),
                    request(primary, "two"),
                    request(switched, "three"),
                    request(primary, "four"),
                ],
            }
        )
    return result


class FakeProcess:
    def __init__(self, output, payload: bytes) -> None:
        self.output = output
        self.payload = payload

    def wait(self, *, timeout: float) -> int:
        del timeout
        self.output.write(self.payload)
        self.output.flush()
        return 0


def test_runner_publishes_valid_hash_only_evidence_and_binds_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = tmp_path / "token"
    token.write_text(SESSION_JWT + "\n", encoding="ascii")
    script = tmp_path / "smoke.cjs"
    script.write_text("console.log('{}')\n", encoding="ascii")
    output = tmp_path / "browser.json"
    model_id = "ullm-qwen3.5-9b-aq4"
    switch_model_id = "llama-qwen3.5-9b-ud-q4"
    payload = (json.dumps(evidence(model_id, switch_model_id)) + "\n").encode("ascii")
    commands: list[list[str]] = []

    def fake_popen(command, *, stdout, **kwargs):
        del kwargs
        commands.append(command)
        return FakeProcess(stdout, payload)

    monkeypatch.setattr(TOOL.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        TOOL,
        "_validate_manifest_identity",
        lambda _path, _model: {"manifest_sha256": "m" * 64},
    )
    result = TOOL.execute(
        output=output,
        manifest=tmp_path / "manifest.json",
        openwebui_session_token_file=token,
        browser_image="sha256:" + "a" * 64,
        openwebui_url="http://127.0.0.1:3000/",
        model_id=model_id,
        model_name="uLLM Qwen3.5 9B AQ4",
        switch_model_id=switch_model_id,
        switch_model_name="llama.cpp Qwen3.5 9B UD-Q4_K_XL",
        browser_script=script,
    )

    assert result["provider_request_count"] == 4
    assert json.loads(output.read_text(encoding="ascii"))["model_id_sha256"] == digest(
        model_id
    )
    assert commands
    assert "--network=host" in commands[0]
    assert "NODE_PATH=/usr/src/app/node_modules" in commands[0]
    assert f"ULLM_MODEL_ID={model_id}" in commands[0]
    assert f"OPENWEBUI_SWITCH_MODEL_ID={switch_model_id}" in commands[0]


def test_runner_publishes_gate_eligible_evidence_without_a_provider_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = tmp_path / "token"
    token.write_text(SESSION_JWT + "\n", encoding="ascii")
    script = tmp_path / "smoke.cjs"
    script.write_text("console.log('{}')\n", encoding="ascii")
    output = tmp_path / "browser.json"
    model_id = "ullm-qwen3.5-9b-aq4"
    payload = (json.dumps(evidence(model_id)) + "\n").encode("ascii")
    commands: list[list[str]] = []

    def fake_popen(command, *, stdout, **kwargs):
        del kwargs
        commands.append(command)
        return FakeProcess(stdout, payload)

    monkeypatch.setattr(TOOL.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        TOOL,
        "_validate_manifest_identity",
        lambda _path, _model: {"manifest_sha256": "m" * 64},
    )
    result = TOOL.execute(
        output=output,
        manifest=tmp_path / "manifest.json",
        openwebui_session_token_file=token,
        browser_image="sha256:" + "a" * 64,
        openwebui_url="http://127.0.0.1:3000/",
        model_id=model_id,
        model_name="uLLM Qwen3.5 9B AQ4",
        browser_script=script,
    )

    document = json.loads(output.read_text(encoding="ascii"))
    assert result["provider_request_count"] == 2
    assert TOOL._load_validator().validate(output)["gate_eligible"] is True
    assert not (set(document) & TOOL.SWITCH_EVIDENCE_FIELDS)
    assert all("OPENWEBUI_SWITCH_MODEL_" not in part for part in commands[0])


def test_runner_cli_allows_switch_arguments_to_be_omitted(tmp_path: Path) -> None:
    args = TOOL.parse_args(
        [
            "--output",
            str(tmp_path / "browser.json"),
            "--manifest",
            str(tmp_path / "active.json"),
            "--openwebui-session-token-file",
            str(tmp_path / "token"),
            "--browser-image",
            "sha256:" + "a" * 64,
            "--openwebui-url",
            "http://127.0.0.1:3000/",
            "--model-id",
            "ullm-qwen3.5-9b-aq4",
            "--model-name",
            "uLLM Qwen3.5 9B AQ4",
        ]
    )

    assert args.switch_model_id is None
    assert args.switch_model_name is None


def test_runner_rejects_external_model_binding_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = tmp_path / "token"
    token.write_text(SESSION_JWT, encoding="ascii")
    script = tmp_path / "smoke.cjs"
    script.write_text("console.log('{}')\n", encoding="ascii")
    output = tmp_path / "browser.json"
    payload = (json.dumps(evidence("candidate", "switch")) + "\n").encode("ascii")

    def fake_popen(command, *, stdout, **kwargs):
        del command, kwargs
        return FakeProcess(stdout, payload)

    monkeypatch.setattr(TOOL.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        TOOL,
        "_validate_manifest_identity",
        lambda _path, _model: {"manifest_sha256": "m" * 64},
    )
    with pytest.raises(TOOL.SmokeError, match="primary model identity"):
        TOOL.execute(
            output=output,
            manifest=tmp_path / "manifest.json",
            openwebui_session_token_file=token,
            browser_image="sha256:" + "a" * 64,
            openwebui_url="http://127.0.0.1:3000",
            model_id="different",
            model_name="candidate",
            switch_model_id="switch",
            switch_model_name="switch",
            browser_script=script,
        )
    assert not output.exists()


def test_runner_rejects_a_v1_active_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "active-v1.json"
    manifest.write_text(
        json.dumps({"schema_version": "ullm.served_model.v1"}), encoding="ascii"
    )
    with pytest.raises(TOOL.SmokeError, match="not v2"):
        TOOL._validate_manifest_identity(manifest, "ullm-qwen3.5-9b-aq4")


@pytest.mark.parametrize("value", ["browser:latest", "sha256:ABC" + "a" * 61])
def test_runner_requires_immutable_browser_image(value: str) -> None:
    with pytest.raises(TOOL.SmokeError, match="immutable Docker"):
        TOOL._validate_image(value)


def test_runner_validates_explicit_browser_container_user() -> None:
    assert TOOL._validate_container_user("1000:1000") == (1000, 1000)
    with pytest.raises(TOOL.SmokeError, match="UID:GID"):
        TOOL._validate_container_user("root")


def test_runner_rejects_gateway_api_key_as_openwebui_session() -> None:
    with pytest.raises(TOOL.SmokeError, match="not a JWT"):
        TOOL._validate_openwebui_session_token(
            b"gateway-api-key", minimum_validity_seconds=30, now_seconds=1
        )
    TOOL._validate_openwebui_session_token(
        SESSION_JWT.encode("ascii"), minimum_validity_seconds=30, now_seconds=1
    )


def test_alternating_r9700_coordinator_serializes_provider_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions: list[tuple[str, str]] = []

    def service(_systemctl: str, action: str, name: str) -> None:
        actions.append((action, name))

    def wait(_rocm: str, expected: set[str], timeout_seconds: float = 60.0) -> None:
        del timeout_seconds
        actions.append(("gpu", ",".join(sorted(expected))))

    monkeypatch.setattr(TOOL, "_service_command", service)
    monkeypatch.setattr(TOOL, "_wait_for_gpu_owner", wait)
    monkeypatch.setattr(TOOL, "_wait_for_tcp_port", lambda *_args, **_kwargs: None)
    coordinator = TOOL._AlternatingServiceCoordinator(
        "systemctl", "rocm-smi", "ullm-openai.service", "llama-qwen35-udq4.service"
    )

    coordinator.transition("before-switch")
    assert coordinator.owner == "llama"
    coordinator.transition("before-return")
    assert coordinator.owner == "ullm"
    assert actions == [
        ("stop", "ullm-openai.service"),
        ("gpu", ""),
        ("start", "llama-qwen35-udq4.service"),
        ("gpu", "llama-server"),
        ("stop", "llama-qwen35-udq4.service"),
        ("gpu", ""),
        ("start", "ullm-openai.service"),
        ("gpu", "ullm-aq4-worker"),
    ]


def test_gpu_owner_probe_accepts_rocm_no_process_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        TOOL.subprocess,
        "run",
        lambda *args, **kwargs: TOOL.subprocess.CompletedProcess(
            args, 0, stdout="", stderr="WARNING: No JSON data to report.\n"
        ),
    )

    assert TOOL._target_gpu_processes("rocm-smi") == []

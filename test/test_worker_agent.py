"""The ``kirocrew-worker`` agent spec, and the two conductors' work-ledger mounts.

Pins the Phase 2 exit criteria about specs: the worker spec is the default agent's
SUPERSET plus ``@kirocrew-work``, and both conductor specs auto-approve the two
conductor tools while auto-approving neither worker tool.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from kiro_crew import agent
from kiro_crew.agent_files import (
    CONDUCTOR_AGENT_FILENAME,
    OWNED_KIRO_AGENT_FILES,
    PIPELINE_CONDUCTOR_AGENT_FILENAME,
    WORKER_AGENT_FILENAME,
)


@pytest.fixture()
def specs(tmp_path, monkeypatch) -> dict[str, dict[str, Any]]:
    """Install the three specs into a throwaway agents dir and read them back."""
    monkeypatch.setattr(agent, "kiro_agents_dir_path", lambda: tmp_path)
    agent._install_worker_agent()
    agent._install_conductor_agent()
    agent._install_pipeline_conductor_agent()
    return {
        name: json.loads((tmp_path / name).read_text(encoding="utf-8"))
        for name in (
            WORKER_AGENT_FILENAME,
            CONDUCTOR_AGENT_FILENAME,
            PIPELINE_CONDUCTOR_AGENT_FILENAME,
        )
    }


# ── the worker is a superset, not a narrowing ─────────────────────────────


def test_the_worker_spec_carries_every_default_tool(specs):
    """The whole of v2's reversal. A worker writes files, runs builds and drives
    git, so anything a NARROWED spec withheld would be something some work item
    needs — the same defect an omitted ``agent`` produces by handing the child
    ``kirocrew-conductor``, which has no ``fs_write``."""
    worker = specs[WORKER_AGENT_FILENAME]
    default_tools = set(agent.build_agent_config().get("tools") or [])
    assert default_tools, "the default template resolved no tools at all"
    assert default_tools <= set(worker["tools"]), (
        "the worker spec withholds a default tool: "
        f"{sorted(default_tools - set(worker['tools']))}"
    )


def test_the_worker_spec_mounts_the_work_server(specs):
    worker = specs[WORKER_AGENT_FILENAME]
    assert "@kirocrew-work" in worker["tools"]
    entry = worker["mcpServers"]["kirocrew-work"]
    assert entry["args"][-1] == "mcp-work"
    assert "autoApprove" not in entry


def test_the_worker_spec_auto_approves_both_worker_tools_and_neither_conductor_one(specs):
    """A worker that must ask permission to say it is blocked will not say it. And a
    worker holding a conductor grant would be auto-approving a tool whose only
    answer to it is a refusal."""
    allowed = specs[WORKER_AGENT_FILENAME]["allowedTools"]
    assert "@kirocrew-work/work_brief" in allowed
    assert "@kirocrew-work/work_report" in allowed
    assert "@kirocrew-work/work_ledger_read" not in allowed
    assert "@kirocrew-work/work_ledger_record" not in allowed


def test_the_worker_spec_keeps_the_default_grants_it_inherited(specs):
    """Appended, not rewritten: a grant added to the default agent tomorrow reaches
    the worker for free, and a grant the ceiling withholds there stays withheld."""
    worker = specs[WORKER_AGENT_FILENAME]
    default_allowed = set(agent.build_agent_config().get("allowedTools") or [])
    assert default_allowed <= set(worker["allowedTools"])


def test_the_worker_prompt_states_the_reporting_contract(specs):
    prompt = specs[WORKER_AGENT_FILENAME]["prompt"]
    for token in (
        "work_brief",
        "work_report",
        "progress",
        "blocked",
        "question",
        "done",
        "artifacts",
        "acceptance",
    ):
        assert token in prompt, token


def test_the_worker_prompt_says_only_decision_is_an_instruction(specs):
    """The prompt's share of the threat model: a worker reads ONE instruction field
    from its conductor, and everything else it reads is state."""
    prompt = specs[WORKER_AGENT_FILENAME]["prompt"]
    assert "`decision`" in prompt
    lowered = prompt.lower()
    assert "instruction" in lowered
    assert "user message" in lowered


def test_the_worker_prompt_carries_the_verbosity_placeholder(specs):
    """Without the token the dashboard verbosity setting never reaches this agent."""
    assert "{{VERBOSITY_BLOCK}}" in specs[WORKER_AGENT_FILENAME]["prompt"]


def test_the_worker_prompt_says_done_is_a_claim(specs):
    """``work_report`` cannot write a verdict, and the prompt must not imply it can."""
    prompt = specs[WORKER_AGENT_FILENAME]["prompt"]
    assert "claim" in prompt.lower()
    assert "verdict" in prompt.lower()


def test_the_worker_spec_derives_its_kas_permissions_from_the_filtered_grants(specs):
    """Derived rather than restated, so a ceiling that strips a grant strips its
    KAS rule with it."""
    worker = specs[WORKER_AGENT_FILENAME]
    assert worker.get("permissions"), "no KAS policy derived"
    rendered = json.dumps(worker["permissions"])
    for ref in worker["allowedTools"]:
        if ref.startswith("@kirocrew-work/"):
            assert ref.split("/", 1)[1] in rendered, ref


def test_the_worker_filename_is_owned_and_wired(specs, tmp_path):
    assert WORKER_AGENT_FILENAME == "kirocrew-worker.json"
    assert WORKER_AGENT_FILENAME in OWNED_KIRO_AGENT_FILES
    assert (tmp_path / WORKER_AGENT_FILENAME).is_file()
    assert specs[WORKER_AGENT_FILENAME]["name"] == "kirocrew-worker"


def test_the_worker_spec_is_not_written_on_the_gateway_boot_path():
    """``rebuild_agent_config`` is reached synchronously from ``_init_services``
    before the dashboard socket binds, so a spec write there is exactly the "new
    synchronous step before the socket binds" that
    ``no-new-work-on-gateway-boot-path`` forbids. The worker spec is materialized on
    first selection instead."""
    import inspect

    from kiro_crew.agent_files import REQUIRED_KIRO_AGENT_FILES

    src = inspect.getsource(agent.rebuild_agent_config)
    assert "_install_worker_agent()" not in src
    assert WORKER_AGENT_FILENAME not in REQUIRED_KIRO_AGENT_FILES


def test_the_worker_spec_materializes_on_first_selection(tmp_path, monkeypatch):
    """``ensure_agent_materialized`` already sits on the spawn path for precisely
    this reason, and is the only moment the file has to exist: a conductor
    dispatching a worker gets the spec written before the child spawns."""
    monkeypatch.setattr(agent, "kiro_agents_dir_path", lambda: tmp_path)
    target = tmp_path / WORKER_AGENT_FILENAME
    assert not target.exists()

    assert agent.ensure_agent_materialized("kirocrew-worker") is True
    assert target.is_file()
    spec = json.loads(target.read_text(encoding="utf-8"))
    assert spec["name"] == "kirocrew-worker"
    assert "@kirocrew-work" in spec["tools"]


def test_a_spec_that_agrees_with_the_ceiling_is_not_rewritten(tmp_path, monkeypatch):
    """Cheap on the hot path: a read, and no write when the answer has not changed."""
    monkeypatch.setattr(agent, "kiro_agents_dir_path", lambda: tmp_path)
    target = tmp_path / WORKER_AGENT_FILENAME
    agent._install_worker_agent()
    before = target.read_bytes()

    def _fail() -> None:
        raise AssertionError("the spec was rewritten although it already matched")

    monkeypatch.setattr(agent, "_install_worker_agent", _fail)
    assert agent.ensure_agent_materialized("kirocrew-worker") is True
    assert target.read_bytes() == before


def test_a_spec_holding_a_revoked_grant_is_regenerated(tmp_path, monkeypatch):
    """The staleness the EAGER installers cannot have: they re-run every
    ``rebuild_agent_config``, so a tightened ceiling reaches them on the next start. A
    spec materialized once has no such moment, and a revoked auto-approve surviving in
    a file is exactly the bypass ``allowedTools`` filtering exists to prevent —
    that list never reaches the PreToolUse gate."""
    monkeypatch.setattr(agent, "kiro_agents_dir_path", lambda: tmp_path)
    target = tmp_path / WORKER_AGENT_FILENAME
    agent._install_worker_agent()
    granted = json.loads(target.read_text(encoding="utf-8"))["allowedTools"]
    assert "@kirocrew-work/work_report" in granted

    # The operator tightens the ceiling on one verb.
    monkeypatch.setattr(agent, "_may_auto_approve", lambda ref: ref != "@kirocrew-work/work_report")
    assert agent._worker_spec_grants_are_stale(target) is True
    assert agent.ensure_agent_materialized("kirocrew-worker") is True
    after = json.loads(target.read_text(encoding="utf-8"))["allowedTools"]
    assert "@kirocrew-work/work_report" not in after
    assert "@kirocrew-work/work_brief" in after
    # And it is now current, so the next spawn does not rewrite it again.
    assert agent._worker_spec_grants_are_stale(target) is False


def test_an_unreadable_spec_is_treated_as_stale(tmp_path, monkeypatch):
    """Fails toward rewriting: a spec that cannot be inspected cannot be trusted to
    carry the current ceiling."""
    monkeypatch.setattr(agent, "kiro_agents_dir_path", lambda: tmp_path)
    target = tmp_path / WORKER_AGENT_FILENAME
    target.write_text("{ truncated", encoding="utf-8")
    assert agent._worker_spec_grants_are_stale(target) is True
    target.write_text('["not", "a", "mapping"]', encoding="utf-8")
    assert agent._worker_spec_grants_are_stale(target) is True


def test_materialization_never_raises_on_the_spawn_path(tmp_path, monkeypatch):
    """Best-effort by contract — it sits on the spawn hot path."""
    monkeypatch.setattr(agent, "kiro_agents_dir_path", lambda: tmp_path)

    def _boom() -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(agent, "_install_worker_agent", _boom)
    assert agent.ensure_agent_materialized("kirocrew-worker") is False


# ── both conductors mount it, per tool ────────────────────────────────────


@pytest.mark.parametrize("filename", [CONDUCTOR_AGENT_FILENAME, PIPELINE_CONDUCTOR_AGENT_FILENAME])
def test_a_conductor_mounts_the_server_and_grants_only_its_own_half(specs, filename):
    """Per tool rather than whole-server, because the worker half is mounted on the
    same server. Missing these is not an error but a silent approval prompt on every
    patrol cycle, which is why they are asserted."""
    spec = specs[filename]
    assert "@kirocrew-work" in spec["tools"]
    entry = spec["mcpServers"]["kirocrew-work"]
    assert entry["args"][-1] == "mcp-work"
    assert "autoApprove" not in entry
    allowed = spec["allowedTools"]
    assert "@kirocrew-work/work_ledger_read" in allowed
    assert "@kirocrew-work/work_ledger_record" in allowed
    assert "@kirocrew-work/work_brief" not in allowed
    assert "@kirocrew-work/work_report" not in allowed
    # Whole-server auto-approve would grant the worker half by the back door.
    assert "@kirocrew-work" not in allowed


@pytest.mark.parametrize("filename", [CONDUCTOR_AGENT_FILENAME, PIPELINE_CONDUCTOR_AGENT_FILENAME])
def test_a_conductor_still_has_no_file_writing_tool(specs, filename):
    """The property the conductor installers' docstrings argue for, re-asserted here
    because this change edits their ``tools`` lists: mounting the work server must
    not have smuggled a write tool in beside it."""
    tools = specs[filename]["tools"]
    assert "fs_write" not in tools
    assert "code" not in tools


def test_the_grant_tuples_name_the_halves_exactly():
    assert agent._CONDUCTOR_WORK_GRANTS == (
        "@kirocrew-work/work_ledger_read",
        "@kirocrew-work/work_ledger_record",
    )
    assert agent._WORKER_WORK_GRANTS == (
        "@kirocrew-work/work_brief",
        "@kirocrew-work/work_report",
    )
    # Together they cover the server's whole surface and overlap nowhere.
    from kiro_crew import mcp_work

    granted = {
        ref.split("/", 1)[1] for ref in agent._CONDUCTOR_WORK_GRANTS + agent._WORKER_WORK_GRANTS
    }
    assert granted == set(mcp_work.WORK_TOOLS)
    assert not set(agent._CONDUCTOR_WORK_GRANTS) & set(agent._WORKER_WORK_GRANTS)


def test_the_hand_built_entry_carries_the_registry_and_home_pins(monkeypatch):
    """Without ``type: registry`` a registry-mode client silently DROPS the entry, so
    the granted tools never launch and the grant is dead with no local error."""
    monkeypatch.setattr(agent, "_mcp_registry_mode", lambda: True)
    monkeypatch.setattr(agent, "_managed_mcp_env", lambda: {"KIROCREW_HOME": "/tmp/home"})
    entry = agent._managed_opt_in_entry("mcp-work")
    assert entry["type"] == agent._MCP_REGISTRY_TYPE
    assert entry["env"] == {"KIROCREW_HOME": "/tmp/home"}
    assert entry["args"][-1] == "mcp-work"


def test_the_hand_built_entry_is_bare_on_a_default_install(monkeypatch):
    monkeypatch.setattr(agent, "_mcp_registry_mode", lambda: False)
    monkeypatch.setattr(agent, "_managed_mcp_env", lambda: {})
    entry = agent._managed_opt_in_entry("mcp-work")
    assert set(entry) == {"command", "args"}

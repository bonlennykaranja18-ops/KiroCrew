"""Replay recorded ACP frames and check what the dispatch layer makes of them.

The Agent SDK boundary refactors (``docs/request-for-change/rfc-crew-agent-sdk-boundary.md``)
move the frame-to-event translation behind a driver. Every one of them is
supposed to be behaviour-preserving, and nothing here checked that: the dispatch
parsers are covered by unit tests that each assert one field of one frame they
construct inline, so a refactor that changes the SHAPE of a turn -- drops the
refinement event a ``tool_call_update`` emits alongside its result, stops
redacting a tool input, reorders two events -- passes every one of them.

This walks the other way round. ``test/fixtures/acp_frames/<backend>/*.jsonl``
holds real frame sequences per backend, one JSON-RPC frame per line, and
``acp_frame_replay_harness`` feeds each sequence through the same parsers the two
reader loops call. This module compares the whole resulting event stream against
a committed snapshot, so a refactor that changes what a backend's stream becomes
fails here with a diff of the events rather than passing green.

**This module is read-only.** It never writes a snapshot, because a test must not
create files in the repo that outlive the run (AUTOSDE ``no-test-side-effects``,
whose own history is a file a test left at the repo root and shipped to main).
Regenerating a snapshot is ``python3 scripts/update_acp_frame_snapshots.py``, and
the rewrite belongs in the same commit as the behaviour change that caused it, so
a reviewer sees the event diff.

**What this does NOT pin.** ``replay_frames`` mirrors the routing the reader
loops perform -- which parser runs for which action, how the caches thread, and
the order events come out -- rather than calling the loops themselves. The
parsers it calls are the real ones, so a change inside a ``_dispatch`` parser
fails here. A change that moves translation INTO ``AcpRuntime._reader_loop`` or
``AcpClient`` and out of a parser does NOT, and event reordering done there is
exactly the kind of change this corpus was built to catch. The mirror is the
price of having a gate before the driver exists: when the RFC's driver lands, the
harness should be retargeted at the driver's public entry point and the mirrored
routing deleted, which turns this from pinning the parsers into pinning the
boundary. ``HANDLED_ACTIONS`` catches a NEW action; it does not catch a changed
mapping for an existing one.

Two ratchets make it hold for backends added later:

* Every id in ``ACP_BACKENDS_KNOWN`` must have a fixture directory. A new
  provider that skips the corpus fails ``test_every_known_backend_has_fixtures``
  -- it does not skip, because a skip is what let the section 5 MCP-projection
  failure reach a public build twice.
* Every action ``classify_notification`` can return must be handled by
  ``replay_frames``. A new frame class routed in production but not there would
  otherwise be silently invisible to the corpus.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest
from acp_frame_replay_harness import (
    CORPUS,
    HANDLED_ACTIONS,
    RECORDED_KINDS,
    backend_dirs,
    expected_path,
    fixture_files,
    fixture_ids,
    read_fixture,
    replay_frames,
    snapshot_json,
)

from kiro_crew.acp import _frame_record
from kiro_crew.acp import client as client_module
from kiro_crew.acp import runtime as runtime_module
from kiro_crew.acp._dispatch import agent_version_from_init, classify_notification
from kiro_crew.acp._frame_record import fixture_dir_name
from kiro_crew.acp_backends import ACP_BACKEND_KIRO, ACP_BACKENDS_KNOWN

REPO_ROOT = Path(__file__).resolve().parents[1]

_UPDATE_SCRIPT = "scripts/update_acp_frame_snapshots.py"

#: A fixture is read by people, so it stays small enough to read.
_MAX_FRAMES = 50


# ── the snapshot walk ───────────────────────────────────────────────────────


@pytest.mark.parametrize("fixture", fixture_files(), ids=fixture_ids())
def test_replayed_events_match_snapshot(fixture: Path) -> None:
    meta, frames = read_fixture(fixture)
    assert meta.get("recorded") in RECORDED_KINDS, (
        f"{fixture.name}: _meta.recorded is {meta.get('recorded')!r}, "
        f"expected one of {sorted(RECORDED_KINDS)}"
    )
    assert fixture_dir_name(meta.get("backend", "")) == fixture.parent.name, (
        f"{fixture.name}: _meta.backend {meta.get('backend')!r} maps to directory "
        f"{fixture_dir_name(meta.get('backend', ''))!r}, but it sits in {fixture.parent.name!r}"
    )
    assert frames, f"{fixture.name} carries a header but no frames"
    assert (
        len(frames) < _MAX_FRAMES
    ), f"{fixture.name} has {len(frames)} frames; keep a fixture under {_MAX_FRAMES}"

    replayed = replay_frames(frames)
    target = expected_path(fixture)
    assert target.exists(), (
        f"{target.name} is missing. Run `python3 {_UPDATE_SCRIPT}` to write it, "
        "and commit it with the change that produced it."
    )
    assert json.loads(target.read_text(encoding="utf-8")) == replayed, (
        f"the events {fixture.parent.name}/{fixture.name} replays into no longer match "
        f"{target.name}. If the change is intended, re-record with "
        f"`python3 {_UPDATE_SCRIPT}` and put the event diff in the review."
    )


def test_the_snapshot_render_is_byte_stable() -> None:
    """The script's on-disk form must round-trip, or every run reports a diff."""
    for fixture in fixture_files():
        _meta, frames = read_fixture(fixture)
        rendered = snapshot_json(replay_frames(frames))
        assert rendered == expected_path(fixture).read_text(encoding="utf-8"), (
            f"{expected_path(fixture).name} differs from the script's render even though the "
            f"parsed events match -- run `python3 {_UPDATE_SCRIPT}`"
        )


# ── ratchets ────────────────────────────────────────────────────────────────


def test_every_known_backend_has_fixtures() -> None:
    """A backend in ACP_BACKENDS_KNOWN without a corpus directory fails here.

    Deliberately a failure and not a skip: the host contract's own history is
    that a provider landed selectable while appearing in none of the buckets,
    and a skipped test is indistinguishable from a passing one on a dashboard.
    """
    present = {d.name for d in backend_dirs()}
    required = {fixture_dir_name(b) for b in ACP_BACKENDS_KNOWN}
    missing = sorted(required - present)
    assert not missing, (
        f"no frame-replay fixtures for {missing}. Every backend in ACP_BACKENDS_KNOWN "
        "needs test/fixtures/acp_frames/<id>/ with at least one .jsonl -- see that "
        "directory's README.md for how to record one."
    )
    stray = sorted(present - required)
    assert not stray, (
        f"fixture directories {stray} name no backend in ACP_BACKENDS_KNOWN; a removed "
        "backend's corpus should be deleted with it"
    )


def test_every_backend_covers_the_required_frame_kinds() -> None:
    """Each backend's corpus must reach the frame classes a turn is made of.

    Without this a directory holding one text chunk would satisfy the ratchet
    above while locking almost nothing.
    """
    gaps: list[str] = []
    for directory in backend_dirs():
        methods: set[str] = set()
        updates: set[str] = set()
        stop_reasons: set[str] = set()
        saw_init = False
        saw_session = False
        for path in sorted(directory.glob("*.jsonl")):
            _meta, frames = read_fixture(path)
            for frame in frames:
                method = frame.get("method")
                if method:
                    methods.add(method)
                result = frame.get("result")
                if isinstance(result, dict):
                    if agent_version_from_init(result):
                        saw_init = True
                    if "sessionId" in result:
                        saw_session = True
                    if result.get("stopReason"):
                        stop_reasons.add(result["stopReason"])
                params = frame.get("params")
                update = params.get("update") if isinstance(params, dict) else None
                if isinstance(update, dict) and update.get("sessionUpdate"):
                    updates.add(update["sessionUpdate"])

        name = directory.name
        if not saw_init:
            gaps.append(f"{name}: no initialize response (a result carrying agentInfo.version)")
        if not saw_session:
            gaps.append(f"{name}: no session/new response (a result carrying sessionId)")
        if "agent_message_chunk" not in updates:
            gaps.append(f"{name}: no agent_message_chunk turn")
        if "tool_call" not in updates:
            gaps.append(f"{name}: no tool_call update")
        if "tool_call_update" not in updates:
            gaps.append(f"{name}: no tool_call_update (tool result) update")
        if "session/request_permission" not in methods:
            gaps.append(f"{name}: no session/request_permission frame")
        if not stop_reasons:
            gaps.append(f"{name}: no end-of-turn response carrying a stopReason")
    assert not gaps, "corpus coverage gaps:\n" + "\n".join(gaps)


def test_replay_handles_every_action_classify_can_return() -> None:
    """``replay_frames`` must have a branch for every action in production.

    The action list is pinned in the harness rather than imported from
    ``_dispatch`` so this fails when ``classify_notification`` grows a return
    value: a frame class routed in production but unhandled in the replay would
    be invisible to every snapshot.
    """
    body = inspect.getsource(classify_notification)
    returned = set(re.findall(r'return "([a-z_]+)"', body))
    assert returned, "found no return values in classify_notification; this ratchet has gone blind"
    unhandled = sorted(returned - set(HANDLED_ACTIONS))
    assert not unhandled, (
        f"classify_notification can return {unhandled}, which replay_frames does not "
        "handle. Add a branch (and a fixture that reaches it) before the frame class "
        "becomes invisible to the corpus."
    )


def test_the_corpus_is_not_vacuous() -> None:
    """A gate that walks an empty corpus must not pass."""
    files = fixture_files()
    assert len(files) >= len(
        ACP_BACKENDS_KNOWN
    ), f"only {len(files)} fixture file(s) for {len(ACP_BACKENDS_KNOWN)} known backend(s)"
    total_events = 0
    for path in files:
        _meta, frames = read_fixture(path)
        total_events += sum(len(entry.get("events", [])) for entry in replay_frames(frames))
    assert total_events > 0, "the corpus replays into zero events; the parsers are not reached"


def test_a_recording_is_marked_live_or_synthesized_honestly() -> None:
    """Every fixture declares its provenance, and the README says which are which."""
    readme = (CORPUS / "README.md").read_text(encoding="utf-8")
    for path in fixture_files():
        meta, _frames = read_fixture(path)
        assert meta.get("recorded") in RECORDED_KINDS, path.name
        assert meta.get("date"), f"{path.name}: _meta.date is required"
        assert "agent_version" in meta, f"{path.name}: _meta.agent_version is required"
    assert (
        "synthesized" in readme
    ), "the corpus README must state which backends are synthesized rather than captured"


def test_replaying_the_whole_corpus_modifies_no_committed_file() -> None:
    """Replay must leave the corpus byte-identical on disk.

    A pytest-time snapshot-update mode is exactly the shape AUTOSDE
    ``no-test-side-effects`` forbids -- its own history is a file a test left at
    the repo root and shipped to main. So the invariant is checked by OBSERVING
    the corpus rather than by grepping this module for a write call: every
    fixture and every snapshot is stat'd, the full replay runs, and nothing may
    have moved.
    """
    tracked = sorted(CORPUS.rglob("*"))
    before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in tracked if p.is_file()}
    assert before, "the corpus is empty; this check would be vacuous"

    for path in fixture_files():
        _meta, frames = read_fixture(path)
        snapshot_json(replay_frames(frames))

    after = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in tracked if p.is_file()}
    assert after == before, (
        "replaying the corpus changed a file on disk: "
        f"{sorted(str(p) for p in set(before) ^ set(after)) or 'contents differ'}"
    )
    assert (
        REPO_ROOT / _UPDATE_SCRIPT
    ).exists(), f"{_UPDATE_SCRIPT} is missing, so there is no way to regenerate a snapshot"


# ── the recorder that produces a fixture ────────────────────────────────────


class TestFrameRecorder:
    """``record_frame`` is the only production code this corpus adds.

    It sits on the hot path of every frame of every session, so the properties
    worth pinning are the ones whose absence would be a production incident
    rather than a wrong fixture: silent when off, off the event loop when on,
    silent when broken, and scrubbed when it writes.
    """

    @staticmethod
    def _clean(monkeypatch, tmp_path):
        _frame_record._reset_for_tests()
        monkeypatch.delenv(_frame_record.ENV_RECORD_FRAMES, raising=False)
        return tmp_path

    def test_unset_env_writes_nothing(self, monkeypatch, tmp_path):
        """The state of every ordinary run and every CI run."""
        self._clean(monkeypatch, tmp_path)
        assert _frame_record.recording_destination() == ""

    def test_a_blank_destination_is_refused_not_resolved_to_the_cwd(self, monkeypatch, tmp_path):
        """Path("") is the CWD, so a blank dest would write into the checkout.

        pytest's CWD is the repository root, which is how a stray kas.jsonl
        appeared there while this was being written.
        """
        self._clean(monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)
        _frame_record.write_frame("kas", {"jsonrpc": "2.0"}, "")
        _frame_record.write_frame("kas", {"jsonrpc": "2.0"}, "   ")
        assert list(tmp_path.iterdir()) == [], "a blank destination wrote a file"

    def test_blank_env_is_off(self, monkeypatch, tmp_path):
        """A blank value is off, not a directory named "" in the cwd."""
        self._clean(monkeypatch, tmp_path)
        monkeypatch.setenv(_frame_record.ENV_RECORD_FRAMES, "   ")
        assert _frame_record.recording_destination() == ""

    @pytest.mark.asyncio
    async def test_record_frame_does_not_touch_the_disk_when_off(self, monkeypatch, tmp_path):
        """The async entry point must return before doing any work."""
        self._clean(monkeypatch, tmp_path)
        calls: list[tuple] = []
        monkeypatch.setattr(
            _frame_record, "write_frame", lambda *a: calls.append(a)  # pragma: no cover
        )
        await _frame_record.record_frame("kas", {"jsonrpc": "2.0"})
        assert calls == []

    @pytest.mark.asyncio
    async def test_record_frame_offloads_the_write_off_the_event_loop(self, monkeypatch, tmp_path):
        """A filesystem syscall on a reader loop stalls every session on it.

        The write must reach a worker thread, so this asserts it runs on a
        DIFFERENT thread than the loop -- not merely that the bytes landed.
        """
        import threading

        self._clean(monkeypatch, tmp_path)
        monkeypatch.setenv(_frame_record.ENV_RECORD_FRAMES, str(tmp_path))
        loop_thread = threading.get_ident()
        seen: list[int] = []
        real_write = _frame_record.write_frame

        def spy(backend, frame, dest):
            seen.append(threading.get_ident())
            real_write(backend, frame, dest)

        monkeypatch.setattr(_frame_record, "write_frame", spy)
        await _frame_record.record_frame("kas", {"jsonrpc": "2.0", "id": 1})
        assert seen, "the write never ran"
        assert seen[0] != loop_thread, "the write ran on the event loop thread"
        assert json.loads((tmp_path / "kas.jsonl").read_text(encoding="utf-8"))["id"] == 1

    def test_a_frame_lands_in_the_backend_file(self, monkeypatch, tmp_path):
        self._clean(monkeypatch, tmp_path)
        dest = str(tmp_path / "frames")
        _frame_record.write_frame(ACP_BACKEND_KIRO, {"jsonrpc": "2.0", "id": 1}, dest)
        _frame_record.write_frame(ACP_BACKEND_KIRO, {"jsonrpc": "2.0", "id": 2}, dest)
        _frame_record.write_frame("kas", {"jsonrpc": "2.0", "id": 3}, dest)

        # kiro-cli's id is "", so its file is named through POLICY_ID_BY_BACKEND.
        kiro_file = Path(dest) / "kiro.jsonl"
        assert kiro_file.exists(), sorted(p.name for p in Path(dest).iterdir())
        lines = kiro_file.read_text(encoding="utf-8").splitlines()
        assert [json.loads(ln)["id"] for ln in lines] == [1, 2]
        assert json.loads((Path(dest) / "kas.jsonl").read_text(encoding="utf-8"))["id"] == 3

    def test_a_credential_is_scrubbed_before_it_is_written(self, monkeypatch, tmp_path):
        self._clean(monkeypatch, tmp_path)
        secret = "ghp_0123456789abcdefghijklmnopqrstuvwxyzAB"
        _frame_record.write_frame("kas", {"params": {"nested": [{"token": secret}]}}, str(tmp_path))
        written = (tmp_path / "kas.jsonl").read_text(encoding="utf-8")
        assert secret not in written
        assert "REDACTED" in written

    def test_the_recording_home_directory_is_scrubbed(self, monkeypatch, tmp_path):
        self._clean(monkeypatch, tmp_path)
        home = tmp_path / "home" / "someone"
        home.mkdir(parents=True)
        monkeypatch.setattr(_frame_record.Path, "home", staticmethod(lambda: home))
        _frame_record.write_frame("kas", {"params": {"path": f"{home}/notes.md"}}, str(tmp_path))
        written = (tmp_path / "kas.jsonl").read_text(encoding="utf-8")
        assert str(home) not in written
        assert "~/notes.md" in written

    def test_an_unwritable_destination_never_raises(self, monkeypatch, tmp_path):
        """A recorder fault must not reach a reader loop.

        In ``AcpRuntime._reader_loop`` an escaping exception ends EVERY
        multiplexed session on the process, so a bad path has to degrade to a log
        line. Pointing the destination at a FILE makes mkdir fail.
        """
        self._clean(monkeypatch, tmp_path)
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("", encoding="utf-8")
        _frame_record.write_frame("kas", {"jsonrpc": "2.0"}, str(blocker))  # must not raise
        assert blocker.read_text(encoding="utf-8") == ""
        assert _frame_record.recording_destination() == "", "a failure must stand recording down"
        _frame_record._reset_for_tests()

    @pytest.mark.asyncio
    async def test_a_dead_executor_never_raises(self, monkeypatch, tmp_path):
        """An offload that cannot be scheduled must degrade, not kill the reader."""
        self._clean(monkeypatch, tmp_path)
        monkeypatch.setenv(_frame_record.ENV_RECORD_FRAMES, str(tmp_path))

        def boom():
            raise RuntimeError("cannot schedule new futures after shutdown")

        monkeypatch.setattr(_frame_record, "subprocess_executor", boom)
        await _frame_record.record_frame("kas", {"jsonrpc": "2.0"})  # must not raise
        assert _frame_record.recording_destination() == ""
        _frame_record._reset_for_tests()

    def test_an_unknown_backend_still_gets_a_filename(self):
        """A recorder must not crash a reader loop over an unexpected label."""
        assert (
            _frame_record.fixture_dir_name("no-such-backend") == _frame_record.UNKNOWN_BACKEND_DIR
        )

    def test_every_known_backend_maps_to_a_usable_directory_name(self):
        names = {fixture_dir_name(b) for b in ACP_BACKENDS_KNOWN}
        assert len(names) == len(ACP_BACKENDS_KNOWN), f"two backends share a directory: {names}"
        for name in names:
            assert name and name != _frame_record.UNKNOWN_BACKEND_DIR
            assert "/" not in name and "\\" not in name and name not in (".", "..")


def test_both_transports_record_their_inbound_frames() -> None:
    """The hook must sit in BOTH reader loops, or a backend records nothing.

    ``AcpRuntime._reader_loop`` serves kiro-cli and KAS; ``AcpClient._read_message``
    serves Claude Code and Codex. A hook in only one of them silently makes the
    other two backends unrecordable, which is invisible until someone tries to
    refresh their corpus.
    """
    for module in (runtime_module, client_module):
        path = Path(module.__file__ or "")
        source = path.read_text(encoding="utf-8")
        assert "await record_frame(" in source, (
            f"{path.name} does not await record_frame; the frames its backends read "
            "cannot be recorded, or the call blocks the event loop"
        )

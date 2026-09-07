"""Opt-in recorder for raw inbound ACP frames — a no-op unless switched on.

Set ``KIROCREW_ACP_RECORD_FRAMES`` to a directory and every agent->client
JSON-RPC frame the two transports read is appended, one JSON object per line, to
``<dir>/<backend>.jsonl``. Unset — which is every ordinary run, and every CI run
— :func:`record_frame` returns before it awaits anything and touches nothing.

Why it exists: the replay corpus under ``test/fixtures/acp_frames/`` is the
behavioural gate that stops a refactor of the dispatch layer silently changing
what a backend's frames become, and a corpus is only worth trusting if a human
can regenerate it from a real backend rather than hand-writing what they assume
the wire looks like. ``test/fixtures/acp_frames/README.md`` documents the
recording procedure and the review a recording must pass before it is committed.

Three properties this module owes the reader loops that call it, because it sits
on the hot path of every frame of every session:

* **It never blocks the event loop.** ``mkdir`` and an append are filesystem
  syscalls, and both reader loops are ``async``, so the write is handed to
  ``subprocess_executor()`` rather than run inline. A blocking syscall on the
  loop freezes every task on it — the user's turn and the liveness heartbeat —
  until the watchdog kills the process.
* **It never raises.** A recorder fault must not reach a reader loop: in
  ``AcpRuntime._reader_loop`` an exception tears down EVERY multiplexed session
  on that process, which a user sees as an unexplained chat failure. Every
  failure mode degrades to one log line and a permanent stand-down.
* **It redacts before it writes.** Frames carry tool output, prompts and
  transcripts. Recording runs the same
  :func:`kiro_crew.acp._dispatch.redact_text` the dashboard path runs, plus a
  home-directory scrub, so an accidental commit of a raw recording is not an
  accidental commit of a credential. That is a floor and not a guarantee — see
  the README's review step, which is what actually keeps account ids and
  private paths out of the corpus.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from kiro_crew.acp._dispatch import redact_text
from kiro_crew.acp_backends import ACP_BACKEND_KIRO, POLICY_ID_BY_BACKEND
from kiro_crew.executors import subprocess_executor

logger = logging.getLogger(__name__)

#: Directory to append recordings to. Absent or blank disables recording.
ENV_RECORD_FRAMES = "KIROCREW_ACP_RECORD_FRAMES"

#: Fixture directory name for a backend whose id is not in
#: ``POLICY_ID_BY_BACKEND``. Cannot happen for a known id — every member of
#: ``ACP_BACKENDS_KNOWN`` has an entry — but a recorder must not crash a reader
#: loop over an unexpected label, so an unknown one lands somewhere nameable.
UNKNOWN_BACKEND_DIR = "unknown"

#: Set once when recording has failed, so a broken destination costs one log
#: line rather than one per frame for the life of the process.
_stood_down = False


def fixture_dir_name(backend: str) -> str:
    """Return the corpus directory name for a backend id.

    The kiro-cli backend's id is the empty string, which is not a filename, so
    the mapping cannot be identity. It reuses ``POLICY_ID_BY_BACKEND`` — the
    module that already had to give every backend id a writable wire name —
    rather than introducing a second table that could disagree with it.
    """
    if backend == ACP_BACKEND_KIRO:
        return POLICY_ID_BY_BACKEND[ACP_BACKEND_KIRO]
    return POLICY_ID_BY_BACKEND.get(backend) or UNKNOWN_BACKEND_DIR


def recording_destination() -> str:
    """Return the recording directory, or ``""`` when recording is off.

    The whole cost of recording being off: one environment lookup. Read on every
    call rather than cached so there is no stale-state path to reason about.
    """
    if _stood_down:
        return ""
    return os.environ.get(ENV_RECORD_FRAMES, "").strip()


def scrub_frame(frame: dict) -> str:
    """Serialize *frame* to one JSON line with credentials and $HOME removed.

    Redaction runs over the serialized text rather than per string value so a
    secret is caught wherever it sits in the frame — a nested tool result, a
    list of content blocks — without this module having to know the shape of
    every frame kind the backends emit.
    """
    text = json.dumps(frame, ensure_ascii=False, sort_keys=True)
    text = redact_text(text)
    # A recording made on a developer's machine otherwise carries their login
    # name in every file path a tool call touched. Substituted on the JSON text,
    # so the escaped form a path takes inside a JSON string is covered too.
    try:
        home = str(Path.home())
    except (OSError, RuntimeError):
        home = ""
    if home and home not in ("/", ""):
        text = text.replace(json.dumps(home)[1:-1], "~").replace(home, "~")
    return text


def write_frame(backend: str, frame: dict, dest: str) -> None:
    """Append one scrubbed frame to ``<dest>/<backend>.jsonl``.

    Runs on a worker thread, never on the event loop. Swallows every failure:
    the caller is a transport reader whose job is the session, not the
    recording.
    """
    if not dest.strip():
        # Path("") resolves to the CWD, so a blank destination would append the
        # frame to ./<backend>.jsonl wherever the gateway happens to be running.
        # A blank destination means recording is off.
        return
    try:
        directory = Path(dest).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        line = scrub_frame(frame)
        with open(
            directory / f"{fixture_dir_name(backend)}.jsonl",
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(line + "\n")
    except Exception as exc:  # noqa: BLE001 - a recorder must never kill a reader
        _stand_down(exc)


async def record_frame(backend: str, frame: dict) -> None:
    """Record one inbound frame, if a recording directory is set.

    Returns before awaiting anything when ``KIROCREW_ACP_RECORD_FRAMES`` is
    unset, which is the state of every ordinary run. When it is set, the
    filesystem work is offloaded so the reader loop is never blocked on disk.
    """
    dest = recording_destination()
    if not dest:
        return
    try:
        await asyncio.get_running_loop().run_in_executor(
            subprocess_executor(), write_frame, backend, frame, dest
        )
    except Exception as exc:  # noqa: BLE001 - a shut-down pool must not kill a reader
        _stand_down(exc)


def _stand_down(exc: BaseException) -> None:
    """Disable recording for the rest of the process after one failure."""
    global _stood_down
    _stood_down = True
    logger.warning(
        "%s is set but recording failed; frame recording is now off for this " "process: %s",
        ENV_RECORD_FRAMES,
        exc,
    )


def _reset_for_tests() -> None:
    """Clear the stand-down latch. For tests only."""
    global _stood_down
    _stood_down = False

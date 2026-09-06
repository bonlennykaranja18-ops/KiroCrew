"""Lifecycle hooks — the companion's runtime follows the app's enabled state.

The gateway's lifecycle dispatcher calls :func:`on_startup` when the app is
enabled (and at gateway boot if it is already enabled) and :func:`on_shutdown`
when it is disabled or the gateway stops. Both are idempotent, because a
re-enable calls startup again on a process that never restarted.

This is the whole reason enabling works now. The previous manifest ran
``open "$HOME/Applications/Crew Companion.app"`` as an ``onEnable`` script, and
``handle_app_api_proxy`` rolls an enable BACK when that script fails — so on any
machine without that app already present, which is every machine but the
author's, the tile could not be switched on at all. There is nothing here to
launch and therefore nothing to fail: the state the user toggles and the state
the runtime follows are the same state.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from kiro_crew.apps.builtins.crew_companion.appearances import AppearanceStore
from kiro_crew.apps.builtins.crew_companion.store import CompanionStore

logger = logging.getLogger(__name__)

#: Process-wide runtime. A module global rather than something hung off ``ctx``
#: because the routes need to reach the same instance, and the dispatcher hands
#: each hook its own context object.
_store: CompanionStore | None = None


def get_appearances() -> AppearanceStore:
    """The appearance store — the install's ONE shared library.

    Delegates to :mod:`kiro_crew.dashboard.appearances` rather than owning an
    instance of its own. Two reasons, and the second is the load-bearing one:

    * Crews wear packs too, so the library has to be readable while this app is
      disabled. A store built by this hook would exist only between startup and
      shutdown.
    * With two instances there would be two libraries. The gallery would import
      a pack into the app's directory and the crew roster would look in the data
      home's, so a pack the user just installed would be missing from the crew
      editor — and one of the two would silently own the colour maps.

    It is created on first use rather than by the startup hook because listing
    packs is a pure read of a directory: there is nothing to schedule and
    nothing to tear down, and a gallery that works before the first tick is one
    less ordering dependency.
    """
    # circular import, and a HARD one: this package's `__init__` imports
    # `backend/routes.py`, which imports this module — so a dashboard module that
    # imports the shared store gets pulled through the whole app package while it
    # is still initialising, and a module-scope import back into the dashboard
    # fails with "partially initialized module" depending only on which side the
    # interpreter reached first. That is import-ORDER dependent, so it passed
    # locally and broke the macOS gateway suite. At call time every module is
    # loaded and the direction no longer matters.
    from kiro_crew.dashboard.appearances import get_appearance_store

    return get_appearance_store()


def get_store() -> CompanionStore | None:
    """The live runtime, or None when the app is disabled or not yet started.

    Routes use this to answer 503 rather than 500 while the hook has not run —
    "not started yet" and "broken" are different answers and the caller can
    retry only one of them.
    """
    return _store


async def on_startup(ctx: Any) -> None:
    """Load persisted state and start ticking. Idempotent.

    Async because loading touches the disk: awaited off the event loop so
    enabling the app never blocks the gateway on file I/O, the same reason the
    reference builtin's hook is async.
    """
    global _store

    data_dir = Path(ctx.data_dir)

    # NOTHING for the appearance library here. This hook runs inside aiohttp's
    # `runner.setup()`, so every statement in it lands BEFORE the dashboard
    # socket binds — and building the store scans the packs directory and, once
    # per install, migrates the app's old private library. That is exactly the
    # unbounded, maintenance-shaped work the boot path must not carry.
    #
    # It costs nothing to defer, because the store's own accessor builds it on
    # first use and every caller resolves it INSIDE a worker thread, so the
    # first request pays it off the event loop rather than the launch paying it
    # on the loop.

    if _store is not None:
        # Re-enable after a disable: the same instance resumes rather than
        # building a second one that would double-fire every reminder.
        _store.start()
        return

    """
    Push a fire the moment it is queued, instead of waiting out the overlay's poll.

    The desktop app this came from held reminders in its main process and pushed
    them straight to the pet window, so a due reminder appeared within its 1s tick.
    Here the store is in the gateway and the overlay polls over HTTP, which added
    up to the poll interval on top of the tick — the reminder was late by seconds,
    and it showed.

    `ctx.events` is the gateway's app event bus, present only when the manifest
    declares the event under `permissions.events`. When it is absent the store runs
    exactly as before and the poll remains the only path: slower, never broken.
    """
    bus = getattr(ctx, "events", None)

    def announce_fire() -> None:
        if bus is None:
            return
        # No payload: this is a doorbell, not the delivery. The overlay drains
        # /pending itself, which keeps ONE ordering authority (the cursor) instead of
        # a pushed copy that could arrive out of order or duplicate what it polls.
        bus.publish("crew-companion:fire")

    store = CompanionStore(data_dir, on_fire=announce_fire)
    await asyncio.to_thread(store.load)
    store.start()
    _store = store
    logger.info("crew-companion: runtime started (data dir %s)", data_dir)


async def on_shutdown(ctx: Any) -> None:  # noqa: ARG001 — ctx unused, kept for the ABI
    """Stop ticking and flush state to disk. Idempotent.

    The instance is deliberately KEPT rather than dropped, so a disable followed
    by an enable resumes with the accumulated stats and the in-flight break
    schedule instead of starting from zero.
    """
    if _store is not None:
        await asyncio.to_thread(_store.stop)
        logger.info("crew-companion: runtime stopped")


def _reset_for_tests() -> None:
    """Drop the process-global runtime. Tests only."""
    global _store
    if _store is not None:
        _store.stop()
    _store = None

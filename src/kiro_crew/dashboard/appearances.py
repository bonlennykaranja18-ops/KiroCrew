"""The shared appearance library — ONE pack store for the whole install.

Appearance packs started as Crew Companion's private gallery, rooted in that
app's own data directory and reachable only through
``/api/apps/crew-companion/...``. Crews now wear packs too
(``agents.*.avatar = {"kind": "pack", "id": ...}``), and a crew's face must not
depend on whether an app happens to be enabled: with the store owned by the app,
disabling Crew Companion would 403 every pack read and blank the roster.

So the store moves up a level. It is rooted at the data home, this module owns
the single instance, and both the dashboard's ``/api/appearances`` routes and
Crew Companion's own gallery routes read that one instance — which is also what
makes the two surfaces show the SAME library rather than two that silently
diverge.

**Shape.** A lazily-built process-wide instance behind a ``threading.Lock``,
the same shape ``artifacts.get_default_store`` uses for the artifact store and
for the same reason: the first caller may be a request handler, a hook, or a
test, and none of them is a natural owner of construction. It lives under
``dashboard/`` rather than at the package root because the consumers are the
dashboard's route table and one builtin app, and because importing a builtin
app's library from the core package root would be a new dependency edge —
``dashboard/session_directive_apply.py`` already establishes that a dashboard
module may reach into a builtin's library, so this adds no new direction.

**Why the store class stays in the app.** ``AppearanceStore`` is unchanged and
still lives in ``crew_companion/appearances.py``. Moving it would be a large
rename touching every one of its tests for no behavioural gain, and the app is
still the surface that AUTHORS packs (the gallery, the sprite editor, PetDex).
What changed is who owns the instance and where it is rooted.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import threading
from pathlib import Path

from kiro_crew.appearance_packs import safe_pack_id
from kiro_crew.apps.builtins.crew_companion.appearances import (
    PACKS_DIRNAME,
    AppearanceStore,
    _safe_id,
)
from kiro_crew.apps.manager import app_dir
from kiro_crew.config.paths import data_home

logger = logging.getLogger(__name__)

#: The library's directory under the data home. A directory of its own rather
#: than a bare ``appearances/`` at the top level, because the store keeps its
#: colour maps beside the packs and both belong to the same library.
LIBRARY_DIRNAME = "appearance-library"

#: The app whose private library is migrated into the shared one, once.
_LEGACY_APP = "crew-companion"
#: Written into the shared library once EVERY legacy pack has been accounted
#: for. Its presence is what ends the migration for good; its absence is what
#: makes a partial run resumable. A marker file rather than "the library has
#: packs in it", because those two questions have different answers the moment
#: one pack moves and the next one fails.
MIGRATION_DONE_FILENAME = ".legacy-migrated"

_store: AppearanceStore | None = None
_store_lock = threading.Lock()


def library_dir() -> Path:
    """Where the shared library lives.

    Resolved per call, never bound at import: ``data_home()`` honours a
    ``KIROCREW_HOME`` override set after this module was imported, which is what
    keeps a pod, a dev backend and a test from reaching into the real install.
    """
    return data_home() / LIBRARY_DIRNAME


def get_appearance_store() -> AppearanceStore:
    """The process-wide appearance store, built and migrated on first use."""
    global _store
    with _store_lock:
        if _store is None:
            store = AppearanceStore(library_dir())
            _migrate_legacy_library(library_dir())
            # Colours migrate on their OWN call, not as the tail of the pack
            # move. Hung off that move, a transient write failure was permanent:
            # the packs had already landed, so the next start found no legacy
            # pack directories, returned early, and never reached the colour
            # step again — the user's recolourings stayed invisible for good.
            # Its no-overwrite check is what makes running every time safe.
            _migrate_legacy_colours(library_dir())
            store.load()
            _store = store
        return _store


def _legacy_data_dir() -> Path:
    """Crew Companion's own data directory, WITHOUT creating it.

    ``apps.manager.app_data_dir`` mkdirs as a side effect of resolving, so it
    cannot be used to ask a question — probing for a legacy library would create
    the very directory it is checking for on an install that never had the app.
    """
    return app_dir(_LEGACY_APP) / "data"


def _migrate_legacy_library(shared_root: Path) -> None:
    """Move Crew Companion's private packs into the shared library. Once.

    A MOVE (``os.replace``), not a copy: two libraries holding the same pack id
    would drift the moment the user recolours or re-edits one of them, and the
    user has no way to tell which copy the roster is drawing from.

    Three rules make this safe to run on every process start:

    * **A completion MARKER ends it, not "the library has packs in it".** Those
      two tests give different answers the moment one pack moves and the next one
      fails: the library is then non-empty, so an emptiness test skips the
      migration forever and every pack after the failure stays invisible — while
      the failure path's own comment promises a retry. The marker is written only
      once every legacy pack has been moved or found already taken, so a partial
      run is resumed and a finished one is never revisited. Being final is what
      stops a pack the user deliberately deleted from the shared library being
      resurrected later.
    * **Nothing is ever deleted.** A pack that cannot be moved is logged and
      left exactly where it was, and a pack id the store would refuse is
      skipped rather than renamed. The worst outcome is a pack that stays
      invisible until someone looks at the log, never one that is gone.
    * **Idempotent by construction.** After a successful run the legacy
      directory holds no pack directories, so the next call finds nothing to
      move and returns without touching the disk beyond two ``iterdir`` calls.
    """
    if _migration_marked_done(shared_root):
        # Settled once and never revisited. This is what stops a pack the user
        # deliberately DELETED from the shared library being resurrected by a
        # later start, which is the hazard the old "shared library already has
        # packs" test was reaching for.
        return
    legacy_packs = _legacy_data_dir() / PACKS_DIRNAME
    shared_packs = shared_root / PACKS_DIRNAME
    try:
        movable = [
            entry
            for entry in sorted(legacy_packs.iterdir())
            if entry.is_dir() and _safe_id(entry.name) is not None
        ]
    except FileNotFoundError:
        # No legacy app ever existed here. That is an ANSWERED question, not a
        # failed read, so it settles: nothing will ever appear in a directory
        # that is not there, and a later fresh install of the app brings no packs
        # with it. Marking it saves two directory scans on every start.
        _mark_migration_done(shared_root)
        return
    except OSError:
        # Present but unreadable — a permission problem, a stalled mount. NOT
        # marked done: an unreadable directory is not an answered question, and
        # marking it would make a transient read failure permanent.
        return
    if not movable:
        # Nothing left to move — that IS the finished state, so record it.
        _mark_migration_done(shared_root)
        return
    try:
        shared_packs.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("appearance library: cannot create %s: %s", shared_packs, exc)
        return
    moved = 0
    stranded = 0
    for entry in movable:
        target = shared_packs / entry.name
        if target.exists():
            # An id already taken in the shared library. That is an ANSWERED
            # question, not a failure: the shared copy wins and the legacy one is
            # history, so it must not hold the marker open forever.
            logger.warning(
                "appearance library: %s already exists in the shared library; "
                "leaving the legacy copy in place",
                entry.name,
            )
            continue
        try:
            os.replace(entry, target)
        except OSError:
            # Across filesystems ``os.replace`` cannot rename a directory, so
            # fall back to a copy. It lands at a STAGING name first and is
            # renamed into place, because a bare ``copytree`` straight to the
            # target is not atomic: a failure part-way would leave a PARTIAL pack
            # sitting at the real id while the legacy original is still intact.
            # Staging plus a rename means the target either does not exist or is
            # complete.
            staging = shared_packs / f".migrating-{entry.name}-{os.getpid()}"
            try:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)
                shutil.copytree(entry, staging)
                os.replace(staging, target)
            except OSError as exc:
                # Remove only the staging copy. The legacy original is untouched,
                # so the pack is still there to migrate on a later start — and
                # `stranded` is what keeps that promise true, by withholding the
                # completion marker so a later start looks again. Without it the
                # first pack landing was enough to end the migration, and every
                # pack after a failure stayed invisible for good while this very
                # comment claimed otherwise.
                shutil.rmtree(staging, ignore_errors=True)
                logger.warning("appearance library: could not migrate pack %s: %s", entry.name, exc)
                stranded += 1
                continue
            shutil.rmtree(entry, ignore_errors=True)
        moved += 1
    if moved:
        logger.info(
            "appearance library: migrated %d pack(s) from the Crew Companion app "
            "directory into %s",
            moved,
            shared_packs,
        )
    if stranded:
        logger.warning(
            "appearance library: %d pack(s) could not be migrated and are still in "
            "the legacy app directory; the migration will be retried on the next "
            "start",
            stranded,
        )
    else:
        _mark_migration_done(shared_root)


def _migration_marked_done(shared_root: Path) -> bool:
    """Whether the legacy migration has already run to completion."""
    try:
        return (shared_root / MIGRATION_DONE_FILENAME).exists()
    except OSError:
        # Unreadable means unknown, and unknown must not read as done: a
        # migration skipped on a bad answer is one that never happens again.
        return False


def _mark_migration_done(shared_root: Path) -> None:
    """Record that every legacy pack has been moved or accounted for.

    Best-effort. A marker that cannot be written costs one repeated scan on the
    next start, which is cheap and idempotent — every pack has already landed, so
    the retry finds the ids taken and accounts for them the same way.
    """
    try:
        shared_root.mkdir(parents=True, exist_ok=True)
        (shared_root / MIGRATION_DONE_FILENAME).write_text("", "utf-8")
    except OSError as exc:
        logger.warning("appearance library: could not record migration completion: %s", exc)


def _migrate_legacy_colours(shared_root: Path) -> None:
    """Carry the recolourings across with the packs they belong to.

    Left behind, a migrated pack would come back in its author's colours and
    look like the recolouring was lost. Only ever written when the shared
    library has none of its own, and a failure is logged rather than raised:
    colour is recoverable by re-picking it, the art is not.
    """
    name = "crew-companion-colours.json"
    legacy = _legacy_data_dir() / name
    shared = shared_root / name
    try:
        if not legacy.is_file() or shared.exists():
            return
        os.replace(legacy, shared)
    except OSError as exc:
        logger.warning("appearance library: could not migrate colour maps: %s", exc)


async def crews_wearing(pack_id: str) -> list[str]:
    """The crews whose ``avatar`` names *pack_id*, sorted.

    Read under the agents routes' own config lock by the caller — see
    :func:`delete_pack_if_unworn`.
    """
    # circular import: config.loader reaches back into this package through the
    # dashboard handler tree, so the config model is imported at call time.
    from kiro_crew.config.loader import KiroCrewConfig

    cfg = await asyncio.to_thread(KiroCrewConfig.load)
    return sorted(
        name
        for name, agent in cfg.agents.items()
        if agent.avatar.get("kind") == "pack" and agent.avatar.get("id") == pack_id
    )


async def delete_pack_if_unworn(pack_id: str, *, force: bool = False) -> tuple[bool, list[str]]:
    """Delete a custom pack unless a crew still wears it.

    Returns ``(deleted, wearers)``. ``wearers`` is non-empty only when the delete
    was REFUSED, so a caller distinguishes "still worn" from "no such pack" by
    that list rather than by re-deriving it.

    **Why this is shared rather than one route's rule.** Both delete paths now
    reach the SAME library — the dashboard's ``DELETE /api/appearances/{id}`` and
    Crew Companion's own gallery route. Before the library was shared those were
    two separate stores, so the gallery could not touch a pack a crew wore; now
    it can, and a guard on only one of the two doors is not a guard. A gallery
    delete that removed the art out from under a crew would blank that face with
    nothing on screen to explain it.

    Three mechanics carry the guarantee:

    * The in-use read and the delete run under the agents routes' own config
      lock. Without it a crew save landing between them would leave exactly the
      reference the check exists to find.
    * The delete itself goes through ``_drained_to_thread``, so a cancelled
      request cannot release that lock with a directory removal still in flight
      — which would let a concurrent crew save observe a half-deleted library.
    * The pack id is CANONICALIZED once, and the same canonical value drives
      both the wearer lookup and the delete. Without that they disagree: the
      store normalizes an id through ``safe_pack_id`` (which strips whitespace)
      while a raw comparison against the config does not, so a delete for
      ``"aurora "`` found no wearer and then removed ``aurora`` — the guard
      bypassed by a trailing space. An id the store would refuse outright is
      answered as "no such pack" without touching the config at all.
    """
    # circular import: handlers.agents imports this module for the shared store
    # and the delete guard, so the lock and the drained-worker helper it owns can
    # only be reached at call time.
    from kiro_crew.dashboard.handlers.agents import _drained_to_thread, _get_config_lock

    ident = safe_pack_id(pack_id)
    if ident is None:
        return False, []
    store = await asyncio.to_thread(get_appearance_store)
    async with _get_config_lock():
        if not force:
            wearers = await crews_wearing(ident)
            if wearers:
                return False, wearers
        deleted = await _drained_to_thread(store.delete_pack, ident)
    return bool(deleted), []


def _reset_for_tests() -> None:
    """Drop the process-global store so the next call rebuilds it. Tests only."""
    global _store
    _store = None

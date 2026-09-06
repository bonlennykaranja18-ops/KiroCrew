"""The shared appearance library: one store, and a surface that outlives the app.

Packs used to belong to Crew Companion, rooted in that app's data directory and
reachable only through ``/api/apps/crew-companion/...`` behind an
``_require_enabled`` gate. Crews now wear packs, which breaks two assumptions at
once: a crew's face must render while that app is DISABLED, and the gallery and
the crew editor must be looking at the SAME library rather than two that drift.

So these tests hold four things:

* one store instance, shared by the dashboard module and the app's hooks;
* the app's old private library is MOVED into the shared one, once, and nothing
  is ever deleted on failure;
* the dashboard routes answer while the app is off;
* deleting a pack a crew still wears is refused, by name, unless forced.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import types
import unittest.mock
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew.apps.builtins.crew_companion import appearances as ap
from kiro_crew.config.loader import KiroCrewAgentConfig, KiroCrewConfig
from kiro_crew.dashboard import appearances as shared
from kiro_crew.dashboard.handlers import agents as agents_handlers

_WAV = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


@pytest.fixture(autouse=True)
def _fresh_store():
    """No test may inherit another's process-global store."""
    shared._reset_for_tests()
    yield
    shared._reset_for_tests()


@pytest.fixture(autouse=True)
def _owner_caller(monkeypatch):
    monkeypatch.setattr(
        "kiro_crew.dashboard.handlers.source_providers.is_owner_dashboard_request",
        lambda request: True,
    )


def _seed_pack(root: Path, ident="aurora", *, sounds=None, extra=None, states=None):
    """A minimal pack under *root* (a store's data dir), returning its directory."""
    pack = root / ap.PACKS_DIRNAME / ident
    pack.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "meta": {"id": ident, "name": ident.title(), "format": "svg"},
        "states": states if states is not None else {"idle": "idle.svg"},
    }
    if sounds is not None:
        manifest["sounds"] = sounds
    (pack / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    for name in (manifest["states"] or {}).values():
        (pack / name).write_text(f"<svg id='{name}'/>", "utf-8")
    for name, content in (extra or {}).items():
        (pack / name).write_text(content, "utf-8")
    return pack


class TestOneSharedStore:
    def test_the_app_and_the_dashboard_read_the_same_instance(self):
        """Two instances would be two libraries.

        A pack imported through the gallery would then be missing from the crew
        editor, and one of the two would silently own the colour maps.
        """
        from kiro_crew.apps.builtins.crew_companion import hooks

        assert hooks.get_appearances() is shared.get_appearance_store()

    def test_the_store_is_rooted_at_the_data_home_not_the_app_dir(self):
        from kiro_crew.config.paths import data_home

        assert shared.library_dir() == data_home() / shared.LIBRARY_DIRNAME
        shared.get_appearance_store()
        # Rooted where the store itself says packs live, so the two agree.
        assert (shared.library_dir() / ap.PACKS_DIRNAME).exists()

    def test_repeated_calls_return_the_same_object(self):
        assert shared.get_appearance_store() is shared.get_appearance_store()

    def test_probing_for_the_legacy_library_does_not_create_the_app_directory(self):
        """``app_data_dir`` mkdirs as a side effect of resolving.

        Using it to ask a question would create the very directory the migration
        is checking for, on an install that never had the app.
        """
        from kiro_crew.apps.manager import app_dir

        legacy = app_dir("crew-companion")
        assert not legacy.exists()
        shared.get_appearance_store()
        assert not legacy.exists()


class TestMigration:
    @staticmethod
    def _legacy_dir() -> Path:
        from kiro_crew.apps.manager import app_dir

        return app_dir("crew-companion") / "data"

    def test_a_private_pack_is_moved_into_the_shared_library(self):
        legacy = self._legacy_dir()
        _seed_pack(legacy, "aurora")
        store = shared.get_appearance_store()
        assert [p["id"] for p in store.list_packs()] == [ap.DEFAULT_PACK, "aurora"]
        # MOVED, not copied: two libraries holding one id drift the moment the
        # user recolours or re-edits either, with no way to tell them apart.
        assert not (legacy / ap.PACKS_DIRNAME / "aurora").exists()
        assert (shared.library_dir() / ap.PACKS_DIRNAME / "aurora").is_dir()

    def test_the_art_survives_the_move(self):
        _seed_pack(self._legacy_dir(), "aurora")
        detail = shared.get_appearance_store().pack_detail("aurora")
        assert detail is not None
        assert detail["animations"]["idle"]["content"] == "<svg id='idle.svg'/>"

    def test_running_again_is_a_no_op(self):
        _seed_pack(self._legacy_dir(), "aurora")
        shared.get_appearance_store()
        shared._reset_for_tests()
        store = shared.get_appearance_store()
        assert [p["id"] for p in store.list_packs()] == [ap.DEFAULT_PACK, "aurora"]

    def test_a_colliding_id_leaves_the_legacy_copy_alone(self):
        """The shared copy wins where the ids collide, and only there.

        This replaced a blunter rule — "the shared library has any packs, so skip
        everything" — which read the wrong question: the library is non-empty the
        moment ONE pack lands, so that rule ended the migration and stranded every
        pack behind a failure. Collision is per-id, and a free id is still
        migrated.
        """
        legacy = self._legacy_dir()
        _seed_pack(legacy, "aurora")
        _seed_pack(shared.library_dir(), "aurora")
        _seed_pack(legacy, "nebula")
        store = shared.get_appearance_store()
        assert [p["id"] for p in store.list_packs()] == [
            ap.DEFAULT_PACK,
            "aurora",
            "nebula",
        ]
        # The colliding legacy copy is left where it is; nothing overwrote the
        # pack the user already had.
        assert (legacy / ap.PACKS_DIRNAME / "aurora").is_dir()
        # ...and a collision is an ANSWERED question, so it does not hold the
        # migration open forever.
        assert (shared.library_dir() / shared.MIGRATION_DONE_FILENAME).exists()

    def test_an_install_that_never_had_the_app_migrates_nothing(self):
        store = shared.get_appearance_store()
        assert [p["id"] for p in store.list_packs()] == [ap.DEFAULT_PACK]

    def test_a_directory_whose_name_is_not_a_pack_id_is_left_alone(self):
        """A ``.old.<pid>`` backup is not a pack.

        The store's own listing filters those out, and renaming one into the
        shared library would resurrect a stale copy as a real pack.
        """
        legacy = self._legacy_dir()
        _seed_pack(legacy, "aurora")
        stale = legacy / ap.PACKS_DIRNAME / "aurora.old.999"
        stale.mkdir()
        (stale / "manifest.json").write_text("{}", "utf-8")
        shared.get_appearance_store()
        assert stale.is_dir()

    def test_a_pack_that_cannot_be_moved_stays_where_it_is(self, monkeypatch):
        """Nothing is ever deleted on failure.

        The worst outcome is a pack that stays invisible until someone reads the
        log; never one that is gone.
        """
        legacy = self._legacy_dir()
        _seed_pack(legacy, "aurora")

        def _boom(src, dst):
            raise OSError("cross-device")

        monkeypatch.setattr(shared.os, "replace", _boom)
        monkeypatch.setattr(
            shared.shutil, "copytree", lambda *a, **k: (_ for _ in ()).throw(OSError("nope"))
        )
        store = shared.get_appearance_store()
        assert [p["id"] for p in store.list_packs()] == [ap.DEFAULT_PACK]
        assert (legacy / ap.PACKS_DIRNAME / "aurora" / "manifest.json").is_file()

    def test_a_cross_filesystem_move_falls_back_to_copy_then_remove(self, monkeypatch):
        legacy = self._legacy_dir()
        _seed_pack(legacy, "aurora")
        real_replace = shared.os.replace

        def _rename_only_files(src, dst):
            # The staging rename is WITHIN the shared library, so it is a
            # same-filesystem move and still works — only the legacy-to-shared
            # directory rename is the cross-device one.
            if Path(src).is_dir() and not Path(src).name.startswith(".migrating-"):
                raise OSError("cross-device")
            return real_replace(src, dst)

        monkeypatch.setattr(shared.os, "replace", _rename_only_files)
        store = shared.get_appearance_store()
        assert [p["id"] for p in store.list_packs()] == [ap.DEFAULT_PACK, "aurora"]
        assert not (legacy / ap.PACKS_DIRNAME / "aurora").exists()

    def test_the_recolourings_travel_with_the_packs(self):
        """Left behind, a migrated pack returns in its author's colours.

        Which reads as the recolouring having been lost, not as a migration.
        """
        legacy = self._legacy_dir()
        _seed_pack(legacy, "aurora")
        (legacy / "crew-companion-colours.json").write_text(
            json.dumps({"aurora": {"#000000": "#ff0000"}}), "utf-8"
        )
        store = shared.get_appearance_store()
        assert store.colour_map("aurora") == {"#000000": "#ff0000"}

    def test_a_shared_colour_file_is_never_overwritten(self):
        legacy = self._legacy_dir()
        _seed_pack(legacy, "aurora")
        (legacy / "crew-companion-colours.json").write_text(
            json.dumps({"aurora": {"#000000": "#ff0000"}}), "utf-8"
        )
        shared.library_dir().mkdir(parents=True, exist_ok=True)
        (shared.library_dir() / "crew-companion-colours.json").write_text(
            json.dumps({"aurora": {"#000000": "#00ff00"}}), "utf-8"
        )
        store = shared.get_appearance_store()
        assert store.colour_map("aurora") == {"#000000": "#00ff00"}


def _app() -> web.Application:
    from kiro_crew.dashboard import handlers
    from kiro_crew.dashboard.routes import agents as agents_routes

    app = web.Application()
    app["state"] = types.SimpleNamespace(conversation_log=None)
    agents_routes.register(app)
    assert handlers.api_appearances_list is not None
    return app


class TestRoutesDoNotDependOnTheApp:
    @pytest.mark.asyncio
    async def test_the_library_lists_while_the_companion_is_disabled(self, monkeypatch):
        """The whole reason the surface moved.

        Under ``/api/apps/crew-companion`` every handler answers 403 while the
        app is off, which would blank the face of every crew wearing a pack.
        """
        monkeypatch.setattr(
            "kiro_crew.apps.builtins.crew_companion.backend.routes.is_app_enabled",
            lambda _name: False,
        )
        _seed_pack(shared.library_dir(), "aurora")
        async with TestClient(TestServer(_app())) as client:
            resp = await client.get("/api/appearances")
            assert resp.status == 200
            assert [p["id"] for p in (await resp.json())["packs"]] == [
                ap.DEFAULT_PACK,
                "aurora",
            ]


class TestListAndDetail:
    @pytest.mark.asyncio
    async def test_the_builtin_is_listed_first(self):
        _seed_pack(shared.library_dir(), "aurora")
        async with TestClient(TestServer(_app())) as client:
            body = await (await client.get("/api/appearances")).json()
        assert body["packs"][0]["id"] == ap.DEFAULT_PACK

    @pytest.mark.asyncio
    async def test_detail_inlines_the_art(self):
        _seed_pack(shared.library_dir(), "aurora")
        async with TestClient(TestServer(_app())) as client:
            body = await (await client.get("/api/appearances/aurora")).json()
        assert body["animations"]["idle"]["content"] == "<svg id='idle.svg'/>"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ident", ["gone", "dots.are.out", "x" * 70])
    async def test_a_miss_or_a_bad_id_is_the_same_404(self, ident):
        """Telling the two apart would hand a caller probing for traversal a
        signal it does not need, and a deleted pack is the ordinary case."""
        async with TestClient(TestServer(_app())) as client:
            resp = await client.get(f"/api/appearances/{ident}")
            assert resp.status == 404
            assert (await resp.json())["code"] == "pack_not_found"


class TestSlotRoute:
    @pytest.mark.asyncio
    async def test_a_present_slot_is_served_as_svg(self):
        _seed_pack(shared.library_dir(), "aurora")
        async with TestClient(TestServer(_app())) as client:
            resp = await client.get("/api/appearances/aurora/slot/idle")
            assert resp.status == 200
            assert resp.headers["Content-Type"].startswith("image/svg+xml")
            assert resp.headers["X-Resolved-Slot"] == "idle"
            assert await resp.text() == "<svg id='idle.svg'/>"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("asked", "present", "expected"),
        [
            ("working", {"idle": "idle.svg", "working": "working.svg"}, "working"),
            ("working", {"idle": "idle.svg", "loading": "loading.svg"}, "loading"),
            ("working", {"idle": "idle.svg", "thinking": "thinking.svg"}, "thinking"),
            ("working", {"idle": "idle.svg"}, "idle"),
            ("done", {"idle": "idle.svg", "done": "done.svg"}, "done"),
            ("done", {"idle": "idle.svg"}, "idle"),
            ("error", {"idle": "idle.svg", "error": "error.svg"}, "error"),
            ("error", {"idle": "idle.svg"}, "idle"),
        ],
    )
    async def test_the_fallback_chain_is_resolved_server_side(self, asked, present, expected):
        """A client probing four names would spend four requests learning what
        the manifest already says -- and the legacy desktop-era slot names
        (``loading``, ``thinking``) keep such a pack animating."""
        _seed_pack(shared.library_dir(), "aurora", states=present)
        async with TestClient(TestServer(_app())) as client:
            resp = await client.get(f"/api/appearances/aurora/slot/{asked}")
            assert resp.status == 200
            assert resp.headers["X-Resolved-Slot"] == expected

    @pytest.mark.asyncio
    async def test_an_open_ended_random_clip_resolves_to_itself(self):
        """A pack's random clips are named by their author.

        A fixed slot vocabulary would make them unfetchable, so "unknown" means
        "not in this pack, even after fallback".
        """
        pack = shared.library_dir() / ap.PACKS_DIRNAME / "aurora"
        pack.mkdir(parents=True)
        (pack / "manifest.json").write_text(
            json.dumps(
                {
                    "meta": {"id": "aurora", "name": "Aurora"},
                    "states": {"idle": "idle.svg"},
                    "random": {"cartwheel": "cartwheel.svg"},
                }
            ),
            "utf-8",
        )
        (pack / "idle.svg").write_text("<svg/>", "utf-8")
        (pack / "cartwheel.svg").write_text("<svg id='cw'/>", "utf-8")
        async with TestClient(TestServer(_app())) as client:
            resp = await client.get("/api/appearances/aurora/slot/cartwheel")
            assert resp.status == 200
            assert resp.headers["X-Resolved-Slot"] == "cartwheel"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("slot", ["nope", "x" * 65])
    async def test_an_unknown_slot_is_a_404(self, slot):
        _seed_pack(shared.library_dir(), "aurora")
        async with TestClient(TestServer(_app())) as client:
            resp = await client.get(f"/api/appearances/aurora/slot/{slot}")
            assert resp.status == 404
            assert (await resp.json())["code"] == "slot_not_found"

    @pytest.mark.asyncio
    async def test_the_builtin_says_the_client_draws_it_itself(self):
        """A distinct code, because "this pack has no files" is not "missing"."""
        async with TestClient(TestServer(_app())) as client:
            resp = await client.get(f"/api/appearances/{ap.DEFAULT_PACK}/slot/idle")
            assert resp.status == 404
            assert (await resp.json())["code"] == "builtin_no_content"

    @pytest.mark.asyncio
    async def test_a_lottie_slot_is_served_as_json(self):
        _seed_pack(shared.library_dir(), "aurora", states={"idle": "idle.json"})
        async with TestClient(TestServer(_app())) as client:
            resp = await client.get("/api/appearances/aurora/slot/idle")
            assert resp.headers["Content-Type"].startswith("application/json")

    @pytest.mark.asyncio
    async def test_a_sprite_slot_is_decoded_so_an_img_tag_can_use_it(self):
        """The route exists to be an ``<img src>``.

        The store keeps a sheet base64-encoded because its write path is
        text-only; base64 text under an image content type is not an image.
        """
        _seed_pack(shared.library_dir(), "aurora", states={"idle": "idle.png"})
        pack = shared.library_dir() / ap.PACKS_DIRNAME / "aurora"
        (pack / "idle.png").write_text(_b64(_PNG), "utf-8")
        async with TestClient(TestServer(_app())) as client:
            resp = await client.get("/api/appearances/aurora/slot/idle")
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "image/png"
            assert await resp.read() == _PNG

    @pytest.mark.asyncio
    async def test_an_etag_match_answers_304(self):
        _seed_pack(shared.library_dir(), "aurora")
        async with TestClient(TestServer(_app())) as client:
            first = await client.get("/api/appearances/aurora/slot/idle")
            etag = first.headers["ETag"]
            second = await client.get(
                "/api/appearances/aurora/slot/idle", headers={"If-None-Match": etag}
            )
            assert second.status == 304
            assert second.headers["X-Resolved-Slot"] == "idle"


class TestSoundRoute:
    @pytest.mark.asyncio
    async def test_a_cue_is_served_with_its_sniffed_type(self):
        _seed_pack(
            shared.library_dir(),
            "aurora",
            sounds={"done": "done.wav"},
            extra={"done.wav": _b64(_WAV)},
        )
        async with TestClient(TestServer(_app())) as client:
            resp = await client.get("/api/appearances/aurora/sound/done")
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "audio/wav"
            assert await resp.read() == _WAV

    @pytest.mark.asyncio
    async def test_an_absent_cue_is_a_404(self):
        _seed_pack(shared.library_dir(), "aurora")
        async with TestClient(TestServer(_app())) as client:
            resp = await client.get("/api/appearances/aurora/sound/done")
            assert resp.status == 404
            assert (await resp.json())["code"] == "sound_not_found"

    @pytest.mark.asyncio
    async def test_an_oversize_cue_is_not_served(self):
        big = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * (ap.MAX_SOUND_BYTES + 1)
        _seed_pack(
            shared.library_dir(),
            "aurora",
            sounds={"done": "done.wav"},
            extra={"done.wav": _b64(big)},
        )
        async with TestClient(TestServer(_app())) as client:
            resp = await client.get("/api/appearances/aurora/sound/done")
            assert resp.status == 404


class TestImport:
    @staticmethod
    def _bundle(ident="imported"):
        return {
            "kind": "crew-companion-pack",
            "version": 1,
            "id": ident,
            "manifest": {
                "meta": {"id": ident, "name": "Imported"},
                "states": {"idle": "idle.svg"},
            },
            "files": {"idle.svg": "<svg/>"},
        }

    @pytest.mark.asyncio
    async def test_json_body_matches_the_companions_request_shape(self):
        """Same ``{"bundle": ...}`` envelope, so the frontend reuses its client."""
        async with TestClient(TestServer(_app())) as client:
            resp = await client.post("/api/appearances/import", json={"bundle": self._bundle()})
            assert resp.status == 200, await resp.json()
            assert (await resp.json())["id"] == "imported"
        assert shared.get_appearance_store().pack_exists("imported")

    @pytest.mark.asyncio
    async def test_a_multipart_upload_is_accepted(self):
        from aiohttp import FormData

        form = FormData()
        form.add_field(
            "file",
            json.dumps(self._bundle("from-file")).encode("utf-8"),
            filename="pack.json",
            content_type="application/json",
        )
        async with TestClient(TestServer(_app())) as client:
            resp = await client.post("/api/appearances/import", data=form)
            assert resp.status == 200, await resp.text()
        assert shared.get_appearance_store().pack_exists("from-file")

    @pytest.mark.asyncio
    async def test_a_colliding_id_is_refused_rather_than_clobbered(self):
        _seed_pack(shared.library_dir(), "imported")
        async with TestClient(TestServer(_app())) as client:
            resp = await client.post("/api/appearances/import", json={"bundle": self._bundle()})
            assert resp.status == 400
            assert (await resp.json())["code"] == "invalid_bundle"

    @pytest.mark.asyncio
    async def test_a_traversal_id_is_refused(self):
        async with TestClient(TestServer(_app())) as client:
            resp = await client.post(
                "/api/appearances/import", json={"bundle": self._bundle("../../evil")}
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_a_body_that_is_not_a_bundle_is_a_400(self):
        async with TestClient(TestServer(_app())) as client:
            assert (await client.post("/api/appearances/import", data=b"not json")).status == 400


class TestDelete:
    @pytest.fixture()
    def worn(self):
        _seed_pack(shared.library_dir(), "aurora")
        cfg = KiroCrewConfig.load()
        cfg.agents["nova"] = KiroCrewAgentConfig(
            kiro_agent="kirocrew", avatar={"kind": "pack", "id": "aurora"}
        )
        cfg.agents["comet"] = KiroCrewAgentConfig(
            kiro_agent="kirocrew", avatar={"kind": "pack", "id": "aurora"}
        )
        cfg.agents["plain"] = KiroCrewAgentConfig(kiro_agent="kirocrew")
        cfg.save()

    @pytest.mark.asyncio
    async def test_a_pack_nobody_wears_is_deleted(self):
        _seed_pack(shared.library_dir(), "aurora")
        async with TestClient(TestServer(_app())) as client:
            resp = await client.delete("/api/appearances/aurora")
            assert resp.status == 200, await resp.json()
        assert not shared.get_appearance_store().pack_exists("aurora")

    @pytest.mark.asyncio
    async def test_a_worn_pack_is_refused_and_the_crews_are_named(self, worn):
        """A roster of blank faces the user cannot explain is the alternative."""
        async with TestClient(TestServer(_app())) as client:
            resp = await client.delete("/api/appearances/aurora")
            assert resp.status == 409
            body = await resp.json()
            assert body["code"] == "pack_in_use"
            assert body["crews"] == ["comet", "nova"]
        assert shared.get_appearance_store().pack_exists("aurora")

    @pytest.mark.asyncio
    async def test_force_deletes_it_anyway(self, worn):
        """The crews keep a dangling reference and fall back to the ghost, which
        is what an absent pack already means."""
        async with TestClient(TestServer(_app())) as client:
            resp = await client.delete("/api/appearances/aurora?force=1")
            assert resp.status == 200, await resp.json()
        assert not shared.get_appearance_store().pack_exists("aurora")
        assert KiroCrewConfig.load().agents["nova"].avatar == {
            "kind": "pack",
            "id": "aurora",
        }

    @pytest.mark.asyncio
    async def test_the_builtin_cannot_be_deleted(self):
        async with TestClient(TestServer(_app())) as client:
            resp = await client.delete(f"/api/appearances/{ap.DEFAULT_PACK}")
            assert resp.status == 400
            assert (await resp.json())["code"] == "builtin_pack"

    @pytest.mark.asyncio
    async def test_the_builtin_cannot_be_forced_either(self):
        async with TestClient(TestServer(_app())) as client:
            resp = await client.delete(f"/api/appearances/{ap.DEFAULT_PACK}?force=1")
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_a_pack_that_is_not_there_is_a_404(self):
        async with TestClient(TestServer(_app())) as client:
            resp = await client.delete("/api/appearances/gone")
            assert resp.status == 404
            assert (await resp.json())["code"] == "pack_not_found"

    @pytest.mark.asyncio
    async def test_a_crew_wearing_a_DIFFERENT_pack_does_not_block(self):
        _seed_pack(shared.library_dir(), "aurora")
        cfg = KiroCrewConfig.load()
        cfg.agents["nova"] = KiroCrewAgentConfig(
            kiro_agent="kirocrew", avatar={"kind": "pack", "id": "nebula"}
        )
        cfg.save()
        async with TestClient(TestServer(_app())) as client:
            assert (await client.delete("/api/appearances/aurora")).status == 200


class TestOwnerGate:
    """Every route is owner-gated, reads included.

    A pack is user-authored content served back verbatim, and the library
    decides what the roster draws — so a non-owner dashboard session must not be
    able to read it, import into it, or delete from it.
    """

    @pytest.fixture(autouse=True)
    def _not_the_owner(self, monkeypatch):
        monkeypatch.setattr(
            "kiro_crew.dashboard.handlers.source_providers.is_owner_dashboard_request",
            lambda request: False,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("get", "/api/appearances"),
            ("get", "/api/appearances/aurora"),
            ("get", "/api/appearances/aurora/slot/idle"),
            ("get", "/api/appearances/aurora/sound/done"),
            ("post", "/api/appearances/import"),
            ("post", "/api/appearances/petdex/fetch"),
            ("delete", "/api/appearances/aurora"),
        ],
    )
    async def test_a_non_owner_is_refused(self, method, path):
        _seed_pack(shared.library_dir(), "aurora")
        async with TestClient(TestServer(_app())) as client:
            resp = await getattr(client, method)(path, json={})
            assert resp.status == 403


class TestPetdexRoute:
    @pytest.mark.asyncio
    async def test_a_hit_is_handed_back(self, monkeypatch):
        monkeypatch.setattr(
            "kiro_crew.dashboard.handlers.appearances.fetch_petdex_pet",
            lambda raw: {"ok": True, "slug": "kirby", "spriteBase64": _b64(_PNG)},
        )
        async with TestClient(TestServer(_app())) as client:
            resp = await client.post(
                "/api/appearances/petdex/fetch", json={"input": "petdex.dev/pets/kirby"}
            )
            assert resp.status == 200
            assert (await resp.json())["slug"] == "kirby"

    @pytest.mark.asyncio
    async def test_a_miss_is_a_200_carrying_ok_false(self, monkeypatch):
        """A miss or an unreachable registry is what the import dialog shows,
        not a client error — the same contract the app's own route has."""
        monkeypatch.setattr(
            "kiro_crew.dashboard.handlers.appearances.fetch_petdex_pet",
            lambda raw: {"ok": False, "error": "Could not reach PetDex"},
        )
        async with TestClient(TestServer(_app())) as client:
            resp = await client.post("/api/appearances/petdex/fetch", json={"input": "x"})
            assert resp.status == 200
            assert (await resp.json())["ok"] is False

    @pytest.mark.asyncio
    async def test_a_body_that_is_not_an_object_is_treated_as_empty(self, monkeypatch):
        seen: list[object] = []
        monkeypatch.setattr(
            "kiro_crew.dashboard.handlers.appearances.fetch_petdex_pet",
            lambda raw: seen.append(raw) or {"ok": False, "error": "no"},
        )
        async with TestClient(TestServer(_app())) as client:
            assert (await client.post("/api/appearances/petdex/fetch", json=["nope"])).status == 200
            assert (
                await client.post("/api/appearances/petdex/fetch", data=b"not json")
            ).status == 200
        assert seen == ["", ""]


class TestMalformedRequests:
    @pytest.mark.asyncio
    async def test_a_json_array_body_is_not_a_bundle(self):
        async with TestClient(TestServer(_app())) as client:
            resp = await client.post("/api/appearances/import", json=["nope"])
            assert resp.status == 400
            assert (await resp.json())["code"] == "invalid_bundle"

    @pytest.mark.asyncio
    async def test_a_multipart_upload_with_no_parts_is_a_400(self):
        from aiohttp import FormData

        async with TestClient(TestServer(_app())) as client:
            resp = await client.post(
                "/api/appearances/import",
                data=FormData()(),
                headers={"Content-Type": "multipart/form-data; boundary=x"},
            )
            assert resp.status == 400
            assert (await resp.json())["code"] == "invalid_bundle"

    @pytest.mark.asyncio
    async def test_a_multipart_part_that_is_not_json_is_a_400(self):
        from aiohttp import FormData

        form = FormData()
        form.add_field("file", b"not json at all", filename="pack.json")
        async with TestClient(TestServer(_app())) as client:
            resp = await client.post("/api/appearances/import", data=form)
            assert resp.status == 400
            assert (await resp.json())["code"] == "invalid_bundle"

    @pytest.mark.asyncio
    async def test_a_sprite_slot_that_will_not_decode_reads_as_absent(self):
        """The pack lists the slot but there is nothing renderable behind it.

        Which is the same answer as the slot not being there, and a better one
        than handing the browser bytes that are not a PNG.
        """
        _seed_pack(shared.library_dir(), "aurora", states={"idle": "idle.png"})
        pack = shared.library_dir() / ap.PACKS_DIRNAME / "aurora"
        (pack / "idle.png").write_text("not base64 !!", "utf-8")
        async with TestClient(TestServer(_app())) as client:
            resp = await client.get("/api/appearances/aurora/slot/idle")
            assert resp.status == 404
            assert (await resp.json())["code"] == "slot_not_found"


class TestTheBootPathCarriesNothing:
    """Enabling the app must not build the library before the socket binds.

    A builtin's `on_startup` runs inside aiohttp's `runner.setup()`, so anything
    it awaits is paid before the dashboard accepts requests — and building the
    store scans the packs directory and, once per install, migrates the app's old
    private library. Deferring costs nothing: every caller resolves the store
    inside a worker thread, so the first request pays it off the loop.
    """

    @pytest.mark.asyncio
    async def test_on_startup_does_not_build_the_store(self, tmp_path, monkeypatch):
        from kiro_crew.apps.builtins.crew_companion import hooks

        # Patched at the SOURCE module, not on `hooks`: the hook reaches the
        # store through a function-local import (the app package and the
        # dashboard are a hard import cycle), so there is no `hooks`-level name
        # to patch and doing so would make this test vacuous.
        monkeypatch.setattr(
            shared,
            "get_appearance_store",
            lambda: (_ for _ in ()).throw(AssertionError("store built on the boot path")),
        )
        hooks._reset_for_tests()
        try:
            await hooks.on_startup(types.SimpleNamespace(data_dir=str(tmp_path), events=None))
        finally:
            hooks._reset_for_tests()

    @pytest.mark.asyncio
    async def test_the_gallery_still_works_without_the_warm_up(self, monkeypatch):
        """Deferring must not leave the app's own routes without a store."""
        monkeypatch.setattr(
            "kiro_crew.apps.builtins.crew_companion.backend.routes.is_app_enabled",
            lambda _name: True,
        )
        from kiro_crew.apps.builtins.crew_companion import hooks
        from kiro_crew.apps.builtins.crew_companion.backend import routes
        from kiro_crew.apps.builtins.crew_companion.store import CompanionStore

        _seed_pack(shared.library_dir(), "aurora")
        store = CompanionStore(shared.library_dir())
        store.load()
        monkeypatch.setattr(hooks, "_store", store)
        app = web.Application()
        routes.register_routes(app)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/apps/crew-companion/appearances")
            assert resp.status == 200
            assert [p["id"] for p in (await resp.json())["packs"]] == [
                ap.DEFAULT_PACK,
                "aurora",
            ]


class TestBothDeleteDoorsShareTheGuard:
    """The library is shared, so a guard on one door is not a guard.

    Before the store was shared the gallery could not touch a pack a crew wore —
    they were two separate libraries. Now it can, and a gallery delete that
    removed the art out from under a crew would blank that face with nothing on
    screen to explain it.
    """

    @pytest.fixture()
    def enabled(self, monkeypatch):
        monkeypatch.setattr(
            "kiro_crew.apps.builtins.crew_companion.backend.routes.is_app_enabled",
            lambda _name: True,
        )

    @pytest.fixture()
    def gallery(self, monkeypatch):
        from kiro_crew.apps.builtins.crew_companion import hooks
        from kiro_crew.apps.builtins.crew_companion.backend import routes
        from kiro_crew.apps.builtins.crew_companion.store import CompanionStore

        store = CompanionStore(shared.library_dir())
        store.load()
        monkeypatch.setattr(hooks, "_store", store)
        app = web.Application()
        routes.register_routes(app)
        return app

    @pytest.fixture()
    def worn(self):
        _seed_pack(shared.library_dir(), "aurora")
        cfg = KiroCrewConfig.load()
        cfg.agents["nova"] = KiroCrewAgentConfig(
            kiro_agent="kirocrew", avatar={"kind": "pack", "id": "aurora"}
        )
        cfg.save()

    @pytest.mark.asyncio
    async def test_the_gallery_refuses_a_worn_pack_and_names_the_crew(self, enabled, gallery, worn):
        async with TestClient(TestServer(gallery)) as client:
            resp = await client.post(
                "/api/apps/crew-companion/appearances/delete", json={"id": "aurora"}
            )
            assert resp.status == 409
            body = await resp.json()
            assert body["code"] == "pack_in_use"
            assert body["crews"] == ["nova"]
        assert shared.get_appearance_store().pack_exists("aurora")

    @pytest.mark.asyncio
    async def test_the_gallery_still_deletes_a_pack_nobody_wears(self, enabled, gallery):
        _seed_pack(shared.library_dir(), "aurora")
        async with TestClient(TestServer(gallery)) as client:
            resp = await client.post(
                "/api/apps/crew-companion/appearances/delete", json={"id": "aurora"}
            )
            assert resp.status == 200, await resp.json()
        assert not shared.get_appearance_store().pack_exists("aurora")

    @pytest.mark.asyncio
    async def test_the_gallery_still_refuses_the_builtin(self, enabled, gallery):
        async with TestClient(TestServer(gallery)) as client:
            resp = await client.post(
                "/api/apps/crew-companion/appearances/delete", json={"id": ap.DEFAULT_PACK}
            )
            assert resp.status == 400
            assert (await resp.json())["code"] == "pack_not_deletable"

    @pytest.mark.asyncio
    async def test_the_delete_cannot_be_abandoned_mid_removal(self):
        """Cancellation must not release the config lock with a removal in flight.

        A plain `to_thread` raises at the await while the worker keeps deleting,
        so a concurrent crew save could observe a half-deleted library. The
        drained helper keeps the await alive until the worker finishes.
        """
        _seed_pack(shared.library_dir(), "aurora")
        used: list[object] = []
        real = agents_handlers._drained_to_thread

        async def _spy(fn, /, *args):
            used.append(fn)
            return await real(fn, *args)

        with unittest.mock.patch.object(agents_handlers, "_drained_to_thread", _spy):
            deleted, wearers = await shared.delete_pack_if_unworn("aurora")
        assert (deleted, wearers) == (True, [])
        assert used, "the delete did not go through _drained_to_thread"


class TestColourMigrationIsRetried:
    @staticmethod
    def _legacy_dir():
        from kiro_crew.apps.manager import app_dir

        return app_dir("crew-companion") / "data"

    def test_a_failed_colour_move_is_retried_on_the_next_start(self, monkeypatch):
        """Hung off the pack move, a transient failure was permanent.

        The packs had already landed, so the next start found no legacy pack
        directories, returned early, and never reached the colour step again —
        the recolourings stayed invisible for good.
        """
        legacy = self._legacy_dir()
        _seed_pack(legacy, "aurora")
        (legacy / "crew-companion-colours.json").write_text(
            json.dumps({"aurora": {"#000000": "#ff0000"}}), "utf-8"
        )
        real_replace = shared.os.replace

        def _fail_the_colour_file(src, dst):
            if str(src).endswith("crew-companion-colours.json"):
                raise OSError("transient")
            return real_replace(src, dst)

        monkeypatch.setattr(shared.os, "replace", _fail_the_colour_file)
        first = shared.get_appearance_store()
        assert first.pack_exists("aurora")
        assert first.colour_map("aurora") == {}

        monkeypatch.setattr(shared.os, "replace", real_replace)
        shared._reset_for_tests()
        second = shared.get_appearance_store()
        assert second.colour_map("aurora") == {"#000000": "#ff0000"}


class TestUntrustedSvgIsServedInert:
    """A pack's SVG is third-party markup on the dashboard's OWN origin.

    An SVG is XML, not a bitmap: it can carry a `<script>` element. An `<img
    src>` will not run it, but navigating straight to the slot URL renders it as
    a DOCUMENT on an origin that already holds the dashboard's session — and a
    pack arrives by import or PetDex fetch, so its author is not necessarily the
    user. The picture tier already refuses SVG outright for this reason; a pack's
    art has to be SVG, so it is served inert instead.
    """

    @pytest.mark.asyncio
    async def test_svg_carries_the_script_none_policy(self):
        _seed_pack(shared.library_dir(), "aurora")
        pack = shared.library_dir() / ap.PACKS_DIRNAME / "aurora"
        (pack / "idle.svg").write_text(
            "<svg xmlns='http://www.w3.org/2000/svg'><script>fetch('/api/agents')</script></svg>",
            "utf-8",
        )
        async with TestClient(TestServer(_app())) as client:
            resp = await client.get("/api/appearances/aurora/slot/idle")
            assert resp.status == 200
            assert (
                resp.headers["Content-Security-Policy"]
                == "script-src 'none'; style-src 'unsafe-inline'"
            )
            assert resp.headers["X-Content-Type-Options"] == "nosniff"

    @pytest.mark.asyncio
    async def test_the_policy_matches_the_one_untrusted_file_reads_use(self):
        """Two policies for one class of content is a drift waiting to happen."""
        from kiro_crew.dashboard.handlers import appearances as handlers_mod

        source = Path(
            Path(handlers_mod.__file__).resolve().parents[1] / "handlers" / "files.py"
        ).read_text(encoding="utf-8")
        assert f'"{handlers_mod._SVG_CSP}"' in source

    @pytest.mark.asyncio
    async def test_the_policy_rides_the_304_too(self):
        """A revalidated response is still the one the browser renders."""
        _seed_pack(shared.library_dir(), "aurora")
        async with TestClient(TestServer(_app())) as client:
            first = await client.get("/api/appearances/aurora/slot/idle")
            second = await client.get(
                "/api/appearances/aurora/slot/idle",
                headers={"If-None-Match": first.headers["ETag"]},
            )
            assert second.status == 304
            assert second.headers["Content-Security-Policy"] == handlers_svg_csp()

    @pytest.mark.asyncio
    async def test_inert_formats_get_nosniff_without_the_svg_policy(self):
        """`nosniff` is what stops a browser re-deciding what a lottie really is."""
        _seed_pack(shared.library_dir(), "aurora", states={"idle": "idle.json"})
        async with TestClient(TestServer(_app())) as client:
            resp = await client.get("/api/appearances/aurora/slot/idle")
            assert resp.headers["X-Content-Type-Options"] == "nosniff"
            assert "Content-Security-Policy" not in resp.headers

    @pytest.mark.asyncio
    async def test_a_sound_is_served_with_nosniff(self):
        _seed_pack(
            shared.library_dir(),
            "aurora",
            sounds={"done": "done.wav"},
            extra={"done.wav": _b64(_WAV)},
        )
        async with TestClient(TestServer(_app())) as client:
            resp = await client.get("/api/appearances/aurora/sound/done")
            assert resp.headers["X-Content-Type-Options"] == "nosniff"


def handlers_svg_csp() -> str:
    from kiro_crew.dashboard.handlers import appearances as handlers_mod

    return handlers_mod._SVG_CSP


class TestTheDeleteGuardCannotBeBypassedByAnIdVariant:
    """The guard and the store have to agree on what the id IS.

    The store normalizes through `safe_pack_id`, which strips whitespace; a raw
    comparison against the config does not. So a delete for `"aurora "` found no
    wearer and then removed `aurora` — the guard walked past by a trailing space.
    """

    @pytest.fixture()
    def worn(self):
        _seed_pack(shared.library_dir(), "aurora")
        cfg = KiroCrewConfig.load()
        cfg.agents["nova"] = KiroCrewAgentConfig(
            kiro_agent="kirocrew", avatar={"kind": "pack", "id": "aurora"}
        )
        cfg.save()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("variant", ["aurora ", " aurora", "  aurora  "])
    async def test_a_whitespace_variant_still_finds_the_wearer(self, worn, variant):
        deleted, wearers = await shared.delete_pack_if_unworn(variant)
        assert (deleted, wearers) == (False, ["nova"])
        assert shared.get_appearance_store().pack_exists("aurora")

    @pytest.mark.asyncio
    async def test_an_id_the_store_would_refuse_never_reaches_the_config(self):
        """No pack can exist under it, so there is nothing to look up."""
        with unittest.mock.patch.object(
            KiroCrewConfig,
            "load",
            side_effect=AssertionError("config read for an impossible id"),
        ):
            assert await shared.delete_pack_if_unworn("../../etc") == (False, [])


class TestCrossFilesystemMigrationIsAtomic:
    @staticmethod
    def _legacy_dir():
        from kiro_crew.apps.manager import app_dir

        return app_dir("crew-companion") / "data"

    def test_a_half_finished_copy_never_shadows_the_intact_original(self, monkeypatch):
        """A bare `copytree` to the target is not atomic.

        A failure part-way leaves a PARTIAL pack at the real id while the legacy
        original is still intact — and the next start then sees a non-empty
        shared library, skips the migration for good, and serves the corrupt copy
        forever while the good one is never looked at again.
        """
        legacy = self._legacy_dir()
        _seed_pack(legacy, "aurora")
        real_replace = shared.os.replace
        real_copytree = shared.shutil.copytree

        def _dir_rename_fails(src, dst):
            if Path(src).is_dir() and not Path(src).name.startswith(".migrating-"):
                raise OSError("cross-device")
            return real_replace(src, dst)

        def _copy_dies_part_way(src, dst, *a, **k):
            real_copytree(src, dst, *a, **k)
            raise OSError("disk full half way through")

        monkeypatch.setattr(shared.os, "replace", _dir_rename_fails)
        monkeypatch.setattr(shared.shutil, "copytree", _copy_dies_part_way)
        first = shared.get_appearance_store()
        # Nothing landed at the real id, so the library still looks untouched...
        assert [p["id"] for p in first.list_packs()] == [ap.DEFAULT_PACK]
        packs_dir = shared.library_dir() / ap.PACKS_DIRNAME
        assert not (packs_dir / "aurora").exists()
        # ...the staging copy was cleaned up, and the original is still there.
        assert list(packs_dir.glob(".migrating-*")) == []
        assert (legacy / ap.PACKS_DIRNAME / "aurora" / "manifest.json").is_file()

        # ...so a later start with working storage still migrates it.
        monkeypatch.setattr(shared.shutil, "copytree", real_copytree)
        shared._reset_for_tests()
        second = shared.get_appearance_store()
        assert [p["id"] for p in second.list_packs()] == [ap.DEFAULT_PACK, "aurora"]

    def test_a_staging_directory_is_not_listed_as_a_pack(self):
        """Its name is not a legal pack id, so the listing filters it out."""
        packs = shared.library_dir() / ap.PACKS_DIRNAME
        packs.mkdir(parents=True, exist_ok=True)
        stray = packs / ".migrating-aurora-999"
        stray.mkdir()
        (stray / "manifest.json").write_text('{"meta": {"id": "x", "name": "x"}}', "utf-8")
        assert [p["id"] for p in shared.get_appearance_store().list_packs()] == [ap.DEFAULT_PACK]


class TestAcceptedOwnerDecisionsAreAudited:
    """Half a permission decision in the log is not an audit trail.

    The shared gate records a DENIAL and returns None on success, so an accepted
    decision left no record: a reader could see who was turned away and not who
    got in. The nearest sibling — the owner-gated `GET /api/agents/{name}/avatar`,
    also user-supplied media on this origin — audits its success, so this module
    does too, reads included.
    """

    @pytest.fixture()
    def audited(self, monkeypatch):
        seen: list[dict] = []
        recorder = types.SimpleNamespace(log_api_access=lambda **kw: seen.append(kw))
        monkeypatch.setattr("kiro_crew.dashboard.handlers.appearances._sel", lambda: recorder)
        return seen

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "path", "operation"),
        [
            ("get", "/api/appearances", "appearances.list"),
            ("get", "/api/appearances/aurora", "appearances.detail"),
            ("get", "/api/appearances/aurora/slot/idle", "appearances.slot"),
            ("delete", "/api/appearances/aurora", "appearances.delete"),
        ],
    )
    async def test_a_successful_owner_read_or_write_is_recorded(
        self, audited, method, path, operation
    ):
        _seed_pack(shared.library_dir(), "aurora")
        async with TestClient(TestServer(_app())) as client:
            resp = await getattr(client, method)(path)
            assert resp.status == 200, await resp.text()
        accepted = [e for e in audited if e.get("outcome") == "success"]
        assert operation in {e["operation"] for e in accepted}

    @pytest.mark.asyncio
    async def test_a_failing_audit_never_changes_the_response(self, monkeypatch):
        """An audit must not break the request it describes."""
        _seed_pack(shared.library_dir(), "aurora")
        boom = types.SimpleNamespace(
            log_api_access=lambda **kw: (_ for _ in ()).throw(RuntimeError("sel down"))
        )
        monkeypatch.setattr("kiro_crew.dashboard.handlers.appearances._sel", lambda: boom)
        async with TestClient(TestServer(_app())) as client:
            assert (await client.get("/api/appearances")).status == 200


class TestMigrationResumesAfterAPartialRun:
    """A pack stranded behind a failure must not be invisible for good.

    With two legacy packs, the first landing made the shared library non-empty —
    so an emptiness test skipped the migration forever and the second pack stayed
    stranded, while the failure path's own comment promised a retry.
    """

    @staticmethod
    def _legacy_dir():
        from kiro_crew.apps.manager import app_dir

        return app_dir("crew-companion") / "data"

    def test_the_pack_after_a_failure_is_retried_on_the_next_start(self, monkeypatch):
        legacy = self._legacy_dir()
        _seed_pack(legacy, "aurora")
        _seed_pack(legacy, "nebula")
        real_replace = shared.os.replace

        def _nebula_fails(src, dst):
            if Path(src).name == "nebula":
                raise OSError("transient")
            return real_replace(src, dst)

        monkeypatch.setattr(shared.os, "replace", _nebula_fails)
        monkeypatch.setattr(
            shared.shutil,
            "copytree",
            lambda *a, **k: (_ for _ in ()).throw(OSError("transient")),
        )
        first = shared.get_appearance_store()
        assert [p["id"] for p in first.list_packs()] == [ap.DEFAULT_PACK, "aurora"]
        assert (legacy / ap.PACKS_DIRNAME / "nebula" / "manifest.json").is_file()
        # The marker is withheld while anything is stranded — that is what makes
        # the promised retry real.
        assert not (shared.library_dir() / shared.MIGRATION_DONE_FILENAME).exists()

        monkeypatch.setattr(shared.os, "replace", real_replace)
        shared._reset_for_tests()
        second = shared.get_appearance_store()
        assert [p["id"] for p in second.list_packs()] == [
            ap.DEFAULT_PACK,
            "aurora",
            "nebula",
        ]

    def test_a_completed_migration_is_marked_and_never_revisited(self):
        _seed_pack(self._legacy_dir(), "aurora")
        shared.get_appearance_store()
        assert (shared.library_dir() / shared.MIGRATION_DONE_FILENAME).exists()

    def test_a_deleted_pack_is_not_resurrected_by_a_later_start(self):
        """Finality is the point of the marker.

        Without it, a per-pack retry would re-migrate a pack the user
        deliberately deleted from the shared library.
        """
        _seed_pack(self._legacy_dir(), "aurora")
        store = shared.get_appearance_store()
        assert store.pack_exists("aurora")
        # Put a legacy copy back, as an older install's leftovers would be, and
        # delete the shared one the way a user would.
        _seed_pack(self._legacy_dir(), "aurora")
        assert store.delete_pack("aurora") is True
        shared._reset_for_tests()
        assert not shared.get_appearance_store().pack_exists("aurora")

    def test_an_install_with_no_legacy_app_marks_itself_done(self):
        shared.get_appearance_store()
        assert (shared.library_dir() / shared.MIGRATION_DONE_FILENAME).exists()

    def test_an_unwritable_marker_costs_a_rescan_not_correctness(self, monkeypatch):
        """A marker that cannot be written must not break the migration.

        Every pack has already landed, so the repeated scan finds the ids taken
        and accounts for them the same way.
        """
        _seed_pack(self._legacy_dir(), "aurora")
        monkeypatch.setattr(
            shared.Path,
            "write_text",
            lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")),
        )
        store = shared.get_appearance_store()
        assert store.pack_exists("aurora")


class TestTheAppsOwnRoutesShareTheOwnerGate:
    """A second door into an owner-only library is not owner-only.

    Crew Companion's appearance routes were gated on the app being enabled, which
    was sufficient while the library was that app's private directory. It stopped
    being sufficient when the same store began deciding what the crew roster
    draws: any authenticated app token could then list, import into, or delete
    from the library the dashboard side gates on ownership.
    """

    @pytest.fixture()
    def enabled(self, monkeypatch):
        monkeypatch.setattr(
            "kiro_crew.apps.builtins.crew_companion.backend.routes.is_app_enabled",
            lambda _name: True,
        )

    @pytest.fixture()
    def gallery(self, monkeypatch):
        from kiro_crew.apps.builtins.crew_companion import hooks
        from kiro_crew.apps.builtins.crew_companion.backend import routes
        from kiro_crew.apps.builtins.crew_companion.store import CompanionStore

        store = CompanionStore(shared.library_dir())
        store.load()
        monkeypatch.setattr(hooks, "_store", store)
        app = web.Application()
        routes.register_routes(app)
        return app

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("get", "/api/apps/crew-companion/appearances"),
            ("get", "/api/apps/crew-companion/appearances/detail?id=aurora"),
            ("post", "/api/apps/crew-companion/appearances/import"),
            ("post", "/api/apps/crew-companion/appearances/delete"),
            ("post", "/api/apps/crew-companion/appearances/save"),
            ("post", "/api/apps/crew-companion/petdex/fetch"),
        ],
    )
    async def test_a_non_owner_cannot_reach_the_library_through_the_app(
        self, enabled, gallery, monkeypatch, method, path
    ):
        monkeypatch.setattr(
            "kiro_crew.dashboard.handlers.source_providers.is_owner_dashboard_request",
            lambda request: False,
        )
        _seed_pack(shared.library_dir(), "aurora")
        async with TestClient(TestServer(gallery)) as client:
            resp = await getattr(client, method)(path, json={})
            assert resp.status == 403

    @pytest.mark.asyncio
    async def test_the_owner_still_reaches_it(self, enabled, gallery):
        _seed_pack(shared.library_dir(), "aurora")
        async with TestClient(TestServer(gallery)) as client:
            resp = await client.get("/api/apps/crew-companion/appearances")
            assert resp.status == 200
            assert [p["id"] for p in (await resp.json())["packs"]] == [
                ap.DEFAULT_PACK,
                "aurora",
            ]

    @pytest.mark.asyncio
    async def test_a_disabled_app_answers_app_disabled_even_to_a_non_owner(
        self, gallery, monkeypatch
    ):
        """The enabled check stays FIRST, so a probe cannot tell the two apart."""
        monkeypatch.setattr(
            "kiro_crew.apps.builtins.crew_companion.backend.routes.is_app_enabled",
            lambda _name: False,
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.handlers.source_providers.is_owner_dashboard_request",
            lambda request: False,
        )
        async with TestClient(TestServer(gallery)) as client:
            resp = await client.get("/api/apps/crew-companion/appearances")
            assert resp.status == 403
            assert (await resp.json())["code"] == "app_disabled"

    @pytest.mark.asyncio
    async def test_the_apps_reminder_routes_are_left_alone(self, enabled, gallery, monkeypatch):
        """Only the routes reaching the SHARED library gained the gate.

        A reminder is this app's own per-user data, not the install-wide library,
        so widening its gate would be a behaviour change this round did not need.
        """
        monkeypatch.setattr(
            "kiro_crew.dashboard.handlers.source_providers.is_owner_dashboard_request",
            lambda request: False,
        )
        async with TestClient(TestServer(gallery)) as client:
            assert (await client.get("/api/apps/crew-companion/reminders")).status == 200


class TestAnAuditNeverDecidesTheOperation:
    """An audit describes an operation; it must never decide it.

    A write that audits AFTER its mutation and lets the audit raise turns a
    completed delete into a 500 — the pack is gone and the client is told the
    request failed, so the user retries and is told there is no such pack.
    """

    @pytest.fixture()
    def sel_is_down(self, monkeypatch):
        boom = types.SimpleNamespace(
            log_api_access=lambda **kw: (_ for _ in ()).throw(RuntimeError("sel down"))
        )
        monkeypatch.setattr("kiro_crew.dashboard.handlers.appearances._sel", lambda: boom)

    @pytest.mark.asyncio
    async def test_a_delete_that_succeeded_is_not_reported_as_a_failure(self, sel_is_down):
        _seed_pack(shared.library_dir(), "aurora")
        async with TestClient(TestServer(_app())) as client:
            resp = await client.delete("/api/appearances/aurora")
            assert resp.status == 200, await resp.text()
        assert not shared.get_appearance_store().pack_exists("aurora")

    @pytest.mark.asyncio
    async def test_an_import_that_succeeded_is_not_reported_as_a_failure(self, sel_is_down):
        bundle = {
            "kind": "crew-companion-pack",
            "version": 1,
            "id": "imported",
            "manifest": {
                "meta": {"id": "imported", "name": "Imported"},
                "states": {"idle": "idle.svg"},
            },
            "files": {"idle.svg": "<svg/>"},
        }
        async with TestClient(TestServer(_app())) as client:
            resp = await client.post("/api/appearances/import", json={"bundle": bundle})
            assert resp.status == 200, await resp.text()
        assert shared.get_appearance_store().pack_exists("imported")

    @pytest.mark.asyncio
    async def test_a_petdex_fetch_still_answers(self, sel_is_down, monkeypatch):
        monkeypatch.setattr(
            "kiro_crew.dashboard.handlers.appearances.fetch_petdex_pet",
            lambda raw: {"ok": False, "error": "no"},
        )
        async with TestClient(TestServer(_app())) as client:
            assert (
                await client.post("/api/appearances/petdex/fetch", json={"input": "x"})
            ).status == 200

    @pytest.mark.asyncio
    async def test_every_audit_on_this_surface_goes_through_the_one_chokepoint(self):
        """Pinned structurally, because the defect was one call site out of four.

        Three review rounds each found a different obligation missing from these
        routes, every time because the surface re-derived it by hand. A direct
        ``_sel().log_api_access`` here is that mistake reappearing.
        """
        from kiro_crew.dashboard.handlers import appearances as handlers_mod

        source = Path(handlers_mod.__file__).read_text(encoding="utf-8")
        direct = [line for line in source.splitlines() if "_sel().log_api_access" in line]
        # Exactly one: the call inside `_audit` itself.
        assert len(direct) == 1, direct


class TestNoImportCycleBetweenTheAppAndTheDashboard:
    """Either side must be importable FIRST.

    `crew_companion/__init__.py` imports `backend/routes.py` at module scope — the
    `register_routes` re-export the gateway's startup registration looks for — so
    any dashboard module that imports the appearance store pulls that routes file
    in while it is still initialising. A module-scope import of the dashboard side
    from there therefore fails with "partially initialized module" depending only
    on which side the interpreter reaches first, which is why it passed locally and
    broke the macOS gateway suite.

    A same-process import proves nothing once either module is already in
    `sys.modules`, so each order runs in a FRESH interpreter.
    """

    _APP = "kiro_crew.apps.builtins.crew_companion"
    _DASH_STORE = "kiro_crew.dashboard.appearances"
    _DASH_HANDLERS = "kiro_crew.dashboard.handlers.appearances"

    @staticmethod
    def _import_in_a_fresh_interpreter(first: str, second: str, tmp_path: Path) -> None:
        """Import *first* then *second* in a subprocess; fail with its stderr."""
        repo_root = Path(__file__).resolve().parents[1]
        script = tmp_path / "probe.py"
        script.write_text(
            "import importlib, sys\n"
            f"importlib.import_module({first!r})\n"
            f"importlib.import_module({second!r})\n"
            "sys.stdout.write('ok')\n",
            encoding="utf-8",
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "src")
        # cwd inside tmp_path: a child inherits pytest's CWD otherwise, and any
        # file it writes would land in the checkout.
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            env=env,
            timeout=120,
        )
        assert (
            result.returncode == 0 and result.stdout.strip() == "ok"
        ), f"importing {first} before {second} failed:\n{result.stderr}"

    def test_the_app_package_can_be_imported_first(self, tmp_path):
        self._import_in_a_fresh_interpreter(self._APP, self._DASH_HANDLERS, tmp_path)

    def test_the_dashboard_handlers_can_be_imported_first(self, tmp_path):
        self._import_in_a_fresh_interpreter(self._DASH_HANDLERS, self._APP, tmp_path)

    def test_the_shared_store_can_be_imported_first(self, tmp_path):
        self._import_in_a_fresh_interpreter(self._DASH_STORE, self._APP, tmp_path)

    def test_the_app_can_be_imported_before_the_shared_store(self, tmp_path):
        self._import_in_a_fresh_interpreter(self._APP, self._DASH_STORE, tmp_path)

    def test_the_apps_routes_module_reaches_the_dashboard_only_at_call_time(self):
        """Pinned structurally, because the failure is import-ORDER dependent.

        The behavioural probes above catch it only in the orders they try; this
        catches a module-scope dashboard import returning to that file at all.
        """
        from kiro_crew.apps.builtins.crew_companion.backend import routes as routes_mod

        source = Path(routes_mod.__file__).read_text(encoding="utf-8")
        module_scope_dashboard_imports = [
            line
            for line in source.splitlines()
            if line.startswith(("from kiro_crew.dashboard", "import kiro_crew.dashboard"))
        ]
        assert module_scope_dashboard_imports == [], module_scope_dashboard_imports

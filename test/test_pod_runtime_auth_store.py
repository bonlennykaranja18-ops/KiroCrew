"""The pod stages the agent runtime's OWN identity store, not just the SSO cache.

`test_the_agent_runtime_auth_stores_stay_visible` in the sandbox-mask suite pins
`.local/share/kiro-cli` and `.local/share/amazon-q` out of every masking tier
because "the agent runtime is itself spawned inside this sandbox and resolves its
own access token from that store". Not masking it is only half the requirement:
under the pod's remapped ``HOME`` the store must also EXIST there. Staging only
`.aws/sso/cache` left the pod's child at kiro-cli's login gate with a readable,
correctly-unmasked corridor -- the cache is not where the token is resolved.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from kiro_crew import pinned_fs
from kiro_crew import sandbox as sb
from kiro_crew.identity_stores import store_mappings
from kiro_crew.pod.runtime import (
    _RUNTIME_AUTH_STORE_FILE_CAP,
    _runtime_auth_store_mappings,
    _stage_runtime_auth_store,
)


def _kiro_mapping(host: Path, platform: str = "linux", environ: dict | None = None):
    # The kiro-cli mapping for *platform*, straight from the authoritative table.
    return next(
        m for m in store_mappings(platform, host, environ or {}) if m.product.value == "kiro-cli"
    )


_OS_HOME_PARTS = ("os-home",)

#: The staging path writes through PINNED no-follow descriptors, so its POSIX
#: behaviour is only observable where the platform provides them. Gated on the
#: CAPABILITY (the same predicate production consults), never on a bare platform
#: name, and the no-capability contract is asserted unconditionally below.
requires_pinned_walk = pytest.mark.skipif(
    not pinned_fs.supports_pinned_walk(),
    reason="staging requires O_DIRECTORY/O_NOFOLLOW + dir_fd; contract asserted separately",
)


def test_a_platform_without_pinned_walk_stages_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The platform-honest contract, asserted on EVERY platform.

    Windows has no ``O_DIRECTORY``/``O_NOFOLLOW``/``dir_fd``, and this staging moves
    sign-in material, so there is no by-name fallback to degrade to: it stages
    nothing and reports 0. Forced through the documented seam so the branch is
    covered on POSIX too, rather than only running where it happens to be true.
    """
    monkeypatch.setattr(pinned_fs, "supports_pinned_walk", lambda: False)
    host = tmp_path / "host-home"
    store = host / ".local" / "share" / "kiro-cli"
    store.mkdir(parents=True)
    (store / "data.sqlite3").write_text("token-bearing store")
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: host))
    os_home = tmp_path / "pod" / "os-home"
    os_home.mkdir(parents=True)

    assert _stage_runtime_auth_store(os_home, _kiro_mapping(host)) == 0
    assert not (os_home / ".local").exists()


@pytest.fixture
def host_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fixture host home with a kiro-cli store shaped like the real one."""
    home = tmp_path / "host-home"
    store = home / ".local" / "share" / "kiro-cli"
    store.mkdir(parents=True)
    (store / "data.sqlite3").write_text("token-bearing store")
    (store / "tui.js").write_text("non-credential asset")
    nested = store / "cache"
    nested.mkdir()
    (nested / "nested-token.json").write_text("nested")
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    return home


@requires_pinned_walk
def test_the_runtime_auth_store_is_staged_into_the_pod_home(
    tmp_path: Path, host_home: Path
) -> None:
    """The regression: without this the child has no token to resolve."""
    os_home = tmp_path / "pod" / "os-home"
    os_home.mkdir(parents=True)

    staged = _stage_runtime_auth_store(os_home, _kiro_mapping(host_home))

    target = os_home / ".local" / "share" / "kiro-cli"
    assert staged == 3
    assert (target / "data.sqlite3").read_text() == "token-bearing store"
    # Nested levels are mirrored too: the store's internal layout is the runtime's
    # contract, and guessing a filename is what made the SSO staging a no-op.
    assert (target / "cache" / "nested-token.json").read_text() == "nested"


@requires_pinned_walk
def test_staged_files_and_directories_get_owner_only_modes(tmp_path: Path, host_home: Path) -> None:
    os_home = tmp_path / "pod" / "os-home"
    os_home.mkdir(parents=True)

    _stage_runtime_auth_store(os_home, _kiro_mapping(host_home))

    target = os_home / ".local" / "share" / "kiro-cli"
    assert oct(target.stat().st_mode)[-3:] == "700"
    assert oct((target / "data.sqlite3").stat().st_mode)[-3:] == "600"


@requires_pinned_walk
def test_an_absent_store_is_not_an_error(tmp_path: Path, host_home: Path) -> None:
    """Best-effort: a host without the store still boots a signed-out pod."""
    os_home = tmp_path / "pod" / "os-home"
    os_home.mkdir(parents=True)

    assert (
        _stage_runtime_auth_store(
            os_home,
            next(
                m for m in store_mappings("linux", host_home, {}) if m.product.value == "amazon-q"
            ),
        )
        == 0
    )


@requires_pinned_walk
def test_existing_pod_files_are_not_clobbered(tmp_path: Path, host_home: Path) -> None:
    """Create-only, so a pod that refreshed its own credential keeps it."""
    os_home = tmp_path / "pod" / "os-home"
    target = os_home / ".local" / "share" / "kiro-cli"
    target.mkdir(parents=True)
    (target / "data.sqlite3").write_text("pod's own refreshed token")

    _stage_runtime_auth_store(os_home, _kiro_mapping(host_home))

    assert (target / "data.sqlite3").read_text() == "pod's own refreshed token"


def test_the_mapping_set_is_derived_from_the_store_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pinned to its source. The earlier revision named the two POSIX
    # .local/share paths literally, so a macOS host -- or one with a redirected
    # XDG_DATA_HOME -- staged NOTHING, and the viability probe then ACCEPTED the
    # signed-out pod that produced, because signed-out is a legitimate boot state.
    home = Path("/home/user")
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))

    assert _runtime_auth_store_mappings() == store_mappings(sys.platform, home, os.environ)
    assert _RUNTIME_AUTH_STORE_FILE_CAP > 0


@requires_pinned_walk
@pytest.mark.parametrize(
    ("platform", "relative"),
    [
        pytest.param(
            "darwin",
            ("Library", "Application Support", "kiro-cli"),
            id="macos",
        ),
        pytest.param("linux", (".local", "share", "kiro-cli"), id="posix"),
    ],
)
def test_a_per_platform_host_store_is_staged(
    tmp_path: Path, platform: str, relative: tuple[str, ...]
) -> None:
    # Each platform's real layout stages from the table, not from a POSIX guess.
    #
    # Two independent axes, which is what made this fail as ``[macos] 0 == 1`` on a
    # runner that is neither macOS nor Linux. The MAPPING's platform decides the
    # source and staged layout and is supplied explicitly here, so this test is
    # about the table's rows rather than the host's identity. Whether staging can
    # HAPPEN at all is a host CAPABILITY: ``_stage_runtime_auth_store`` refuses and
    # returns 0 where ``pinned_fs.supports_pinned_walk()`` is False (Windows has no
    # ``O_DIRECTORY``/``O_NOFOLLOW``/``dir_fd``), because the pod's own credential
    # tree is published through the pinned no-follow chokepoint and degrading that
    # to a by-name write is not on offer. So the platform ROWS are parametrized and
    # the CAPABILITY is gated -- the no-capability half is asserted unconditionally
    # by ``test_a_platform_without_pinned_walk_stages_nothing``, so nothing about
    # this contract goes unproven on a host that skips this case.
    host = tmp_path / f"host-{platform}"
    (host / Path(*relative)).mkdir(parents=True)
    (host / Path(*relative) / "data.sqlite3").write_text("token")
    os_home = tmp_path / f"pod-{platform}" / "os-home"
    os_home.mkdir(parents=True)

    assert _stage_runtime_auth_store(os_home, _kiro_mapping(host, platform)) == 1
    assert (os_home / Path(*relative) / "data.sqlite3").read_text() == "token"


@requires_pinned_walk
def test_an_xdg_redirected_source_is_followed(tmp_path: Path) -> None:
    # The table honours XDG_DATA_HOME on the SOURCE side; staging follows it.
    host = tmp_path / "host"
    redirected = tmp_path / "elsewhere" / "share"
    (redirected / "kiro-cli").mkdir(parents=True)
    (redirected / "kiro-cli" / "data.sqlite3").write_text("token")
    os_home = tmp_path / "pod" / "os-home"
    os_home.mkdir(parents=True)

    mapping = _kiro_mapping(host, "linux", {"XDG_DATA_HOME": str(redirected)})
    assert _stage_runtime_auth_store(os_home, mapping) == 1
    # Staged at the FIXED default layout, which is where the child will look.
    assert (os_home / ".local" / "share" / "kiro-cli" / "data.sqlite3").exists()


def test_the_staged_store_is_readable_through_the_pod_mask_in_every_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verified, not assumed: no tier re-anchors a mask over the staged store.

    The tier lists exclude `.local/share/kiro-cli` by design, so
    `_pod_os_home_targets` should never produce an entry covering it. Asserted for
    all three tiers because staging a token into a bind-masked-empty directory is
    exactly the failure mode this PR already hit once with `.aws`.
    """
    os_home = "/pods/p1/os-home"
    monkeypatch.setenv("KIROCREW_POD", "1")
    monkeypatch.setenv("KIROCREW_OS_HOME", os_home)

    for dirs in (
        tuple(sb._sandbox_policy().strict_dirs()),
        tuple(sb._sandbox_policy().cc_dirs()),
        tuple(sb._STANDARD_DIRS),
    ):
        targets = _pod_targets = sb._pod_os_home_targets(dirs)
        for mapping in store_mappings("linux", Path("/home/user"), {}):
            path = os.path.join(os_home, *mapping.staged_relative.parts)
            blocked = [t for t in _pod_targets if path == t or path.startswith(t + os.sep)]
            assert blocked == [], f"{path} masked by {blocked} (tier had {len(targets)} entries)"

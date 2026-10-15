"""Certificate dependencies must remain readable inside an otherwise empty root."""

from pathlib import Path

import pytest

from atrex_runtime.workers import filesystem


@pytest.fixture
def trust_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    certs = tmp_path / "etc/pki/tls/certs"
    certs.mkdir(parents=True)
    monkeypatch.setattr(filesystem, "_CERTIFICATE_DIRECTORIES", (str(certs),))
    monkeypatch.setattr(filesystem, "_CERTIFICATE_BUNDLES", ())
    return certs


def test_certificate_mounts_complete_external_chain_without_parent_exposure(
    tmp_path: Path, trust_store: Path,
) -> None:
    target = tmp_path / "etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem"
    target.parent.mkdir(parents=True)
    target.write_text("public CA")
    private = target.parent / "private.key"
    private.write_text("MUST_NOT_BE_VISIBLE")
    intermediate = target.parent.parent / "bundle.pem"
    intermediate.symlink_to("pem/tls-ca-bundle.pem")
    (trust_store / "ca-bundle.crt").symlink_to(intermediate)
    (trust_store / "abcdef01.0").symlink_to("ca-bundle.crt")

    mounts = filesystem.certificate_mounts()

    assert mounts == [
        "--ro-bind", str(trust_store), str(trust_store),
        "--ro-bind", str(target), str(target),
        "--symlink", str(target), str(intermediate),
    ]
    assert str(private) not in mounts
    assert str(target.parent) not in mounts
    assert str(tmp_path / "etc/pki") not in mounts


def test_certificate_directory_alias_preserves_external_file_links(
    tmp_path: Path, trust_store: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias = tmp_path / "certs-alias"
    alias.symlink_to(trust_store, target_is_directory=True)
    target = tmp_path / "public-ca.crt"
    target.write_text("public CA")
    (trust_store / "root.pem").symlink_to(target)
    monkeypatch.setattr(filesystem, "_CERTIFICATE_DIRECTORIES", (str(alias),))

    assert filesystem.certificate_mounts() == [
        "--ro-bind", str(trust_store), str(trust_store),
        "--symlink", str(trust_store), str(alias),
        "--ro-bind", str(target), str(target),
    ]


def test_missing_broken_cyclic_and_directory_links_are_not_added(
    tmp_path: Path, trust_store: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (trust_store / "missing.pem").symlink_to(tmp_path / "missing")
    (trust_store / "cycle.pem").symlink_to("cycle.pem")
    private = tmp_path / "private"
    private.mkdir()
    (trust_store / "directory").symlink_to(private, target_is_directory=True)
    monkeypatch.setattr(filesystem, "_CERTIFICATE_BUNDLES", (str(tmp_path / "absent.pem"),))

    assert filesystem.certificate_mounts() == [
        "--ro-bind", str(trust_store), str(trust_store),
    ]


def test_system_mounts_include_standalone_ca_bundle_without_its_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "etc/ssl/cert.pem"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("public CA")
    monkeypatch.setattr(filesystem, "_SYSTEM_PATHS", ())
    monkeypatch.setattr(filesystem, "_CERTIFICATE_DIRECTORIES", ())
    monkeypatch.setattr(filesystem, "_CERTIFICATE_BUNDLES", (str(bundle),))

    assert filesystem.system_mounts() == ["--ro-bind", str(bundle), str(bundle)]

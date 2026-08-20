"""Registry schema-version acceptance tests."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from conftest import digest, seed_lineage

from atrex_runtime.domain.errors import InvalidTransitionError
from atrex_runtime.domain.ids import parse_epoch_id
from atrex_runtime.domain.models import ChallengerProposalType
from atrex_runtime.registry.sqlite import SCHEMA_VERSION, SqliteRegistry


def test_registry_initializes_current_schema(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    with SqliteRegistry(path):
        pass
    with closing(sqlite3.connect(path)) as connection:
        row = connection.execute("PRAGMA user_version").fetchone()

    assert row == (SCHEMA_VERSION,)


def test_registry_migrates_schema_14_kernel_catalog_measurements(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE kernel_revisions(id TEXT PRIMARY KEY)")
        connection.execute("PRAGMA user_version = 14")

    with SqliteRegistry(path):
        pass

    with closing(sqlite3.connect(path)) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()
        measurement_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'kernel_measurements'"
        ).fetchone()

    assert version == (SCHEMA_VERSION,)
    assert measurement_table == ("kernel_measurements",)


def test_registry_migrates_schema_15_with_stable_baseline_version(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    with SqliteRegistry(path) as registry:
        seeded = seed_lineage(registry)

    with closing(sqlite3.connect(path)) as connection:
        connection.execute("DROP TABLE lineage_agent_versions")
        connection.execute("DROP TABLE lineage_kernel_versions")
        connection.execute("PRAGMA user_version = 15")

    with SqliteRegistry(path) as registry:
        catalog = registry.list_lineage_kernels(seeded.lineage_id)
        agents = registry.list_lineage_agent_revisions(seeded.lineage_id)

    assert len(catalog) == 1
    assert catalog[0].revision.id == seeded.baseline.id
    assert catalog[0].revision_number == 0
    assert len(agents) == 1
    assert agents[0].revision.id == seeded.active_revision_id
    assert agents[0].revision_number == 0


def test_registry_migrates_schema_16_with_stable_agent_version(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    with SqliteRegistry(path) as registry:
        seeded = seed_lineage(registry)

    with closing(sqlite3.connect(path)) as connection:
        connection.execute("DROP TABLE lineage_agent_versions")
        connection.execute("PRAGMA user_version = 16")

    with SqliteRegistry(path) as registry:
        agents = registry.list_lineage_agent_revisions(seeded.lineage_id)

    assert len(agents) == 1
    assert agents[0].revision.id == seeded.active_revision_id
    assert agents[0].revision_number == 0
    assert agents[0].disposition == "baseline"


def test_registry_migrates_schema_18_with_immediate_challenger_default(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE lineages(id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO lineages VALUES ('lineage-old')")
        connection.execute("PRAGMA user_version = 18")
        connection.commit()

    with SqliteRegistry(path):
        pass

    with closing(sqlite3.connect(path)) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()
        start_epoch = connection.execute(
            "SELECT challenger_start_epoch FROM lineages WHERE id = 'lineage-old'"
        ).fetchone()

    assert version == (SCHEMA_VERSION,)
    assert start_epoch == (1,)


def test_registry_migrates_schema_19_challenger_provenance(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    epoch_id = "epoch_0123456789abcdef0123456789abcdef"
    active_id = "agentrev_00000000000000000000000000000000"
    challenger_id = "agentrev_11111111111111111111111111111111"
    trace = digest("legacy-evolution-trace")
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            """
            CREATE TABLE epochs(id TEXT PRIMARY KEY);
            CREATE TABLE kernel_agent_revisions(
                id TEXT PRIMARY KEY,
                parent_id TEXT,
                evolution_trace_digest TEXT
            );
            CREATE TABLE epoch_challengers(
                epoch_id TEXT NOT NULL,
                challenger_ordinal INTEGER NOT NULL,
                kernel_agent_revision_id TEXT NOT NULL,
                PRIMARY KEY(epoch_id, challenger_ordinal)
            );
            """
        )
        connection.execute("INSERT INTO epochs VALUES (?)", (epoch_id,))
        connection.execute(
            "INSERT INTO kernel_agent_revisions VALUES (?, NULL, NULL)",
            (active_id,),
        )
        connection.execute(
            "INSERT INTO kernel_agent_revisions VALUES (?, ?, ?)",
            (challenger_id, active_id, trace),
        )
        connection.execute(
            "INSERT INTO epoch_challengers VALUES (?, 1, ?)",
            (epoch_id, challenger_id),
        )
        connection.execute("PRAGMA user_version = 19")
        connection.commit()

    with SqliteRegistry(path) as registry:
        challenger = registry.list_epoch_challengers(parse_epoch_id(epoch_id))[0]

    assert challenger.proposal_type is ChallengerProposalType.EVOLVED
    assert challenger.base_revision_id == active_id
    assert challenger.evolution_trace_digest == trace


def test_registry_migrates_schema_20_with_default_agent_models(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            """
            CREATE TABLE campaigns(id TEXT PRIMARY KEY);
            CREATE TABLE lineages(id TEXT PRIMARY KEY);
            PRAGMA user_version = 20;
            """
        )

    with SqliteRegistry(path):
        pass

    with closing(sqlite3.connect(path)) as connection:
        campaign_columns = {row[1] for row in connection.execute("PRAGMA table_info(campaigns)")}
        lineage_columns = {row[1] for row in connection.execute("PRAGMA table_info(lineages)")}

    assert "problem_generalization_model" in campaign_columns
    assert {"optimizer_model", "evolver_model"} <= lineage_columns


def test_registry_migrates_schema_21_to_lineage_seed_agent_creator(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    with SqliteRegistry(path) as registry:
        seed_lineage(registry)

    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA user_version = 21")

    with SqliteRegistry(path):
        pass
    with closing(sqlite3.connect(path)) as connection:
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'kernel_agent_revisions'"
        ).fetchone()

    assert row is not None
    assert "lineage_seed" in row[0]


def test_registry_migrates_schema_22_with_campaign_evolver_commit(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            """
            CREATE TABLE campaigns(id TEXT PRIMARY KEY);
            PRAGMA user_version = 22;
            """
        )

    with SqliteRegistry(path):
        pass
    with closing(sqlite3.connect(path)) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(campaigns)")}

    assert "evolver_commit" in columns


def test_registry_binds_legacy_evolver_commit_once(tmp_path: Path) -> None:
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        seeded = seed_lineage(registry)
        campaign_id = registry.get_lineage(seeded.lineage_id).campaign_id

        bound = registry.ensure_campaign_evolver_commit(campaign_id, "a" * 40)

        assert bound.evolver_commit == "a" * 40
        assert registry.ensure_campaign_evolver_commit(campaign_id, "a" * 40) == bound
        with pytest.raises(InvalidTransitionError, match="freezes Evolver commit"):
            registry.ensure_campaign_evolver_commit(campaign_id, "b" * 40)


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_registry_rejects_pre_release_old_schema(tmp_path: Path, version: int) -> None:
    path = tmp_path / "registry.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(f"PRAGMA user_version = {version}")

    with pytest.raises(RuntimeError, match=f"no migration path from Registry schema {version}"):
        SqliteRegistry(path)


def test_registry_closes_connection_when_initialization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = sqlite3.connect(tmp_path / "captured.sqlite")

    def connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        return connection

    def fail_migration(_self: SqliteRegistry) -> None:
        raise RuntimeError("broken")

    monkeypatch.setattr(sqlite3, "connect", connect)
    monkeypatch.setattr(SqliteRegistry, "_migrate", fail_migration)

    with pytest.raises(RuntimeError, match="broken"):
        SqliteRegistry(tmp_path / "registry.sqlite")

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")

"""Direction ancestry is validated, append-only and independent of measurement truth."""

import pytest

from atrex_runtime.direction_genealogy import (
    RELATIONSHIPS,
    relationship_fields,
    validate_relationship,
)

A = "direction_" + "a" * 32
B = "direction_" + "b" * 32
C = "direction_" + "c" * 32
X = "experiment_" + "1" * 32
Y = "experiment_" + "2" * 32


@pytest.mark.parametrize("kind", RELATIONSHIPS)
def test_validates_all_relationships_with_direction_and_experiment_parents(kind: str) -> None:
    value = validate_relationship(
        C,
        {
            "relationship": kind,
            "derived_from_direction_ids": [A, B],
            "derived_from_experiment_ids": [X, Y],
        },
        {A: {}, B: {}},
        {X: {"direction_id": A}, Y: {"direction_id": B}},
    )
    assert value["relationship"] == kind
    assert value["derived_from_direction_ids"] == [A, B]
    assert value["derived_from_experiment_ids"] == [X, Y]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"relationship": "retry"}, "at least one parent"),
        ({"derived_from_direction_ids": [A]}, "relationship must"),
        ({"relationship": "unknown", "derived_from_direction_ids": [A]}, "relationship must"),
        ({"relationship": "retry", "derived_from_direction_ids": [C]}, "not visible"),
        ({"relationship": "retry", "derived_from_experiment_ids": [Y]}, "not visible"),
        ({"relationship": "retry", "derived_from_direction_ids": [A, A]}, "duplicate"),
        ({"relationship": "retry", "derived_from_experiment_ids": [X, X]}, "duplicate"),
        ({"relationship": "retry", "derived_from_direction_ids": ["invalid"]}, "invalid"),
        (
            {
                "relationship": "combination",
                "derived_from_direction_ids": [A],
                "derived_from_experiment_ids": [X],
            },
            "two distinct",
        ),
        (
            {
                "relationship": "port",
                "derived_from_direction_ids": [A],
                "supersedes_direction_id": A,
            },
            "requires correction",
        ),
        (
            {
                "relationship": "correction",
                "derived_from_direction_ids": [A],
                "supersedes_direction_id": B,
            },
            "parent Directions",
        ),
    ],
)
def test_rejects_invalid_ancestry(value: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_relationship(C, value, {A: {}, B: {}}, {X: {"direction_id": A}})


def test_rejects_self_reference_and_transitive_cycle() -> None:
    with pytest.raises(ValueError, match="cycles"):
        validate_relationship(
            C, {"relationship": "retry", "derived_from_direction_ids": [C]}, {C: {}}, {}
        )
    with pytest.raises(ValueError, match="cycles"):
        validate_relationship(
            C,
            {"relationship": "retry", "derived_from_direction_ids": [A]},
            {A: {"derived_from_experiment_ids": [X]}, C: {}},
            {X: {"direction_id": C}},
        )


def test_supersedes_does_not_modify_parent_or_fabricate_legacy_links() -> None:
    parent = {"status": "abandoned"}
    value = validate_relationship(
        C,
        {
            "relationship": "correction",
            "derived_from_experiment_ids": [X],
            "supersedes_direction_id": A,
        },
        {A: parent},
        {X: {"direction_id": A}},
    )
    assert value["supersedes_direction_id"] == A
    assert parent == {"status": "abandoned"}
    assert validate_relationship(B, {}, {A: parent}, {}) == {}
    assert relationship_fields({"relationship": None, "derived_from_direction_ids": []}) == {}

"""Direction ancestry is validated, append-only and independent of measurement truth."""

import pytest

from atrex_runtime.direction_genealogy import (
    RELATIONSHIPS,
    relationship_fields,
    suggestion_availability,
    validate_relationship,
)

A = "direction_" + "a" * 32
B = "direction_" + "b" * 32
C = "direction_" + "c" * 32
X = "experiment_" + "1" * 32
Y = "experiment_" + "2" * 32


@pytest.mark.parametrize("kind", [kind for kind in RELATIONSHIPS if kind != "adoption"])
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


def test_adoption_uses_only_a_suggested_parent() -> None:
    suggested = {"status": "suggested"}
    value = validate_relationship(
        C,
        {"relationship": "adoption", "derived_from_direction_ids": [A]},
        {A: suggested},
        {},
    )
    assert value["derived_from_direction_ids"] == [A]
    assert suggested == {"status": "suggested"}
    with pytest.raises(ValueError, match="unexpired suggested parent"):
        validate_relationship(
            C,
            {"relationship": "adoption", "derived_from_direction_ids": [B]},
            {B: {"status": "completed"}},
            {},
        )
    with pytest.raises(ValueError, match="one suggested parent"):
        validate_relationship(
            C,
            {
                "relationship": "adoption",
                "derived_from_direction_ids": [A],
                "derived_from_experiment_ids": [X],
            },
            {A: suggested},
            {X: {"direction_id": A}},
        )


@pytest.mark.parametrize("status", ["expired", "adopted"])
def test_expired_suggestion_can_be_refined_but_not_adopted(status: str) -> None:
    with pytest.raises(ValueError, match="unexpired suggested parent"):
        validate_relationship(
            C,
            {"relationship": "adoption", "derived_from_direction_ids": [A]},
            {A: {"status": status}},
            {},
        )
    assert validate_relationship(
        C,
        {"relationship": "refinement", "derived_from_direction_ids": [A]},
        {A: {"status": status}},
        {},
    )["derived_from_direction_ids"] == [A]


@pytest.mark.parametrize(
    ("current", "origin", "ttl", "expected"),
    [
        (1, 0, 1, "suggested"),  # Bootstrap is offered during Epoch 1.
        (2, 0, 1, "expired"),
        (2, 2, 1, "suggested"),  # Evolver's new suggestion lasts this Epoch.
        (3, 2, 1, "expired"),
        (2, 1, 2, "suggested"),
        (3, 1, 2, "expired"),
    ],
)
def test_suggestion_availability_by_epoch(
    current: int, origin: int, ttl: int, expected: str
) -> None:
    assert suggestion_availability(current, origin, ttl) == expected

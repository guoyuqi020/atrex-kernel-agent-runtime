"""Resolution of Runtime operator names onto the pinned Store's scope vocabulary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atrex_local_wiki.operator_family import OperatorFamilyResolver

OPERATORS = ("moe", "gemm", "grouped-gemm", "flash-attention", "gdn", "norm")
FAMILIES = ("moe", "gemm", "flash-attention", "gdn", "norm")

DOCUMENTS = (
    ("gdn", "chunk gated delta rule recurrent state update for decode"),
    ("gdn", "gated delta net chunked scan state passing"),
    ("norm", "rmsnorm residual epsilon reduction over the hidden dimension"),
    ("norm", "layernorm welford accumulation single pass"),
    ("moe", "expert routing top-k dispatch and combine"),
    ("gemm", "tile quantization and split-k accumulation"),
)


def _resolver(**kwargs: object) -> OperatorFamilyResolver:
    return OperatorFamilyResolver(
        operators=OPERATORS,
        families=FAMILIES,
        documents=DOCUMENTS,
        **kwargs,  # type: ignore[arg-type]
    )


def test_a_single_name_token_reaches_the_store_operator() -> None:
    scope = _resolver().resolve("fused_moe_fp8")

    assert (scope.axis, scope.value, scope.confidence) == ("operator", "moe", "direct")
    assert scope.is_hard_filter


def test_adjacent_tokens_join_into_a_hyphenated_store_token() -> None:
    scope = _resolver().resolve("flash_attention_v2")

    assert (scope.axis, scope.value) == ("operator", "flash-attention")
    assert scope.confidence == "direct"


def test_the_longest_join_wins_over_the_token_it_contains() -> None:
    scope = _resolver().resolve("grouped_gemm_fp8")

    assert scope.value == "grouped-gemm"


def test_the_corpus_names_the_family_when_no_token_matches() -> None:
    scope = _resolver().resolve("chunk_gated_delta_rule")

    assert (scope.axis, scope.value, scope.confidence) == ("family", "gdn", "voted")
    assert scope.is_hard_filter


def test_an_unclear_vote_leaves_the_scope_open() -> None:
    # Both families use this word equally, so neither can claim the operator.
    documents = (
        ("gemm", "split-k accumulation over tiles"),
        ("moe", "split-k accumulation over experts"),
    )
    resolver = OperatorFamilyResolver(operators=OPERATORS, families=FAMILIES, documents=documents)

    scope = resolver.resolve("accumulation_probe")

    assert scope.confidence == "ambiguous"
    assert scope.value is None
    assert not scope.is_hard_filter


def test_an_unknown_name_resolves_to_nothing() -> None:
    scope = _resolver().resolve("wholly_unrelated_thing")

    assert (scope.value, scope.confidence) == (None, "unresolved")
    assert not scope.is_hard_filter


def test_a_declaration_outranks_the_automatic_stages() -> None:
    scope = _resolver(declared={"fused_moe_fp8": "gemm"}).resolve("fused_moe_fp8")

    assert (scope.axis, scope.value, scope.confidence) == ("operator", "gemm", "declared")


def test_a_declaration_must_name_a_store_token() -> None:
    with pytest.raises(ValueError, match="not Store tokens"):
        _resolver(declared={"fused_moe_fp8": "not-a-store-token"})


def test_terms_drop_precision_and_layout_noise() -> None:
    scope = _resolver().resolve("fused_moe_fp8_v2")

    assert scope.terms == ("moe",)


def test_resolution_is_insensitive_to_name_spelling() -> None:
    resolver = _resolver()

    assert resolver.resolve("Fused-MoE-FP8").value == "moe"
    assert resolver.resolve("fusedMoe_fp8").value is None


def test_an_index_without_records_is_rejected(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"count": 0}), encoding="utf-8")

    with pytest.raises(ValueError, match="no record list"):
        OperatorFamilyResolver.from_index(index)


def test_an_index_builds_both_axes_from_record_scope(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "search_text": "expert routing dispatch",
                        "retrieval": {"scope": {"operator_family": "moe", "operators": ["moe"]}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    resolver = OperatorFamilyResolver.from_index(index)

    assert resolver.resolve("fused_moe_fp8").value == "moe"

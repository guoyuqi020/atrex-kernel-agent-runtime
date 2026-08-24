"""Resolve a Runtime operator name onto the pinned Store's own scope vocabulary.

The Runtime already knows the operator authoritatively, but the Store indexes
knowledge under its own coarser tokens: `fused_moe_fp8` is filed under `moe` and
`chunk_gated_delta_rule` under `gdn`. Comparing the whole name against the
vocabulary never matches either, so the operator axis silently drops out of the
query and unrelated operators' records fill the answer.

Three deterministic stages recover the token without a maintained alias table,
so a new Kernel needs no edit here:

* an explicit declaration, for the rare name the corpus cannot speak for;
* tokenization of the name plus adjacent joins, matched against the vocabulary;
* a vote over the corpus itself, weighting each name token by how rare it is,
  which improves on its own as records are added.

A vote that no family wins by a clear margin resolves to nothing. Guessing a
family is worse than leaving the axis open, because a wrong hard filter hides
the records the caller actually needs.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MAX_JOIN_LENGTH = 3
MIN_TOKEN_LENGTH = 3
VOTE_MARGIN = 1.5
PHRASE_WEIGHT = 2.5

# Words that carry no operator identity. Precision and layout terms are the bulk
# of a Kernel name and would otherwise vote for whichever family happens to
# discuss them most.
_NOISE = frozenset(
    {
        "fused",
        "fusion",
        "kernel",
        "kernels",
        "op",
        "ops",
        "operator",
        "fwd",
        "bwd",
        "forward",
        "backward",
        "fp8",
        "fp16",
        "fp32",
        "bf16",
        "fp4",
        "nvfp4",
        "int8",
        "int4",
        "e4m3",
        "e5m2",
        "v1",
        "v2",
        "v3",
        "dynamic",
        "static",
        "batched",
    }
)

Confidence = Literal["declared", "direct", "voted", "ambiguous", "unresolved"]


@dataclass(frozen=True, slots=True)
class OperatorScope:
    """How one operator name maps onto the Store, and how much to trust it."""

    axis: Literal["operator", "family"] | None
    value: str | None
    confidence: Confidence
    terms: tuple[str, ...]

    @property
    def is_hard_filter(self) -> bool:
        """Whether the mapping is trustworthy enough to narrow the query scope."""
        return self.value is not None and self.confidence in ("declared", "direct", "voted")


def fold(value: str) -> str:
    """Collapse a token to letters and digits, matching the Store's own folding."""
    return re.sub(r"[^a-z0-9]", "", value.strip().lower())


def _tokens(name: str) -> tuple[str, ...]:
    return tuple(part for part in re.split(r"[^A-Za-z0-9]+", name.lower()) if part)


def _candidates(tokens: Iterable[str]) -> list[str]:
    """The name's tokens plus adjacent joins, longest first.

    `flash_attention` reaches the `flash-attention` token only as a join, while
    `fused_moe_fp8` reaches `moe` as a single token. Longest first so a specific
    token such as `grouped-gemm` wins over the `gemm` it contains.
    """
    parts = list(tokens)
    out: list[str] = []
    for size in range(min(MAX_JOIN_LENGTH, len(parts)), 0, -1):
        for start in range(len(parts) - size + 1):
            out.append("-".join(parts[start : start + size]))
    return out


class OperatorFamilyResolver:
    """Map operator names onto Store scope tokens using only the Store's index."""

    def __init__(
        self,
        *,
        operators: Iterable[str],
        families: Iterable[str],
        documents: Iterable[tuple[str, str]],
        declared: Mapping[str, str] | None = None,
    ) -> None:
        self._operators = {fold(value): value for value in operators if value}
        self._families = {fold(value): value for value in families if value}
        self._declared = {fold(key): value for key, value in (declared or {}).items()}
        unknown = sorted(
            value
            for value in self._declared.values()
            if fold(value) not in self._families and fold(value) not in self._operators
        )
        if unknown:
            raise ValueError(f"declared operator families are not Store tokens: {unknown}")
        self._family_size: Counter[str] = Counter()
        self._document_frequency: Counter[str] = Counter()
        self._documents: list[tuple[str, str, frozenset[str]]] = []
        for family, text in documents:
            if not family:
                continue
            words = frozenset(re.findall(r"[a-z0-9]+", text))
            self._documents.append((family, text, words))
            self._family_size[family] += 1
            self._document_frequency.update(words)
        self._total = len(self._documents)

    @classmethod
    def from_index(
        cls,
        index_path: Path,
        *,
        declared: Mapping[str, str] | None = None,
    ) -> OperatorFamilyResolver:
        """Build from the public kernel Store index without importing Store tools."""
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"kernel Store index is unreadable: {index_path}") from error
        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            raise ValueError(f"kernel Store index has no record list: {index_path}")
        operators: set[str] = set()
        families: set[str] = set()
        documents: list[tuple[str, str]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            retrieval = record.get("retrieval")
            scope = retrieval.get("scope") if isinstance(retrieval, dict) else None
            if not isinstance(scope, dict):
                continue
            family = scope.get("operator_family")
            if isinstance(family, str) and family:
                families.add(family)
                text = record.get("search_text")
                documents.append((family, text if isinstance(text, str) else ""))
            for operator in scope.get("operators") or []:
                if isinstance(operator, str) and operator:
                    operators.add(operator)
        return cls(
            operators=operators,
            families=families,
            documents=documents,
            declared=declared,
        )

    def resolve(self, operator: str) -> OperatorScope:
        """Return the Store axis for one operator name, or an open scope."""
        tokens = _tokens(operator)
        terms = tuple(
            token for token in tokens if len(token) >= MIN_TOKEN_LENGTH and token not in _NOISE
        )
        if not tokens:
            return OperatorScope(None, None, "unresolved", ())

        declared = self._declared.get(fold(operator))
        if declared is not None:
            axis: Literal["operator", "family"] = (
                "operator" if fold(declared) in self._operators else "family"
            )
            return OperatorScope(axis, declared, "declared", terms)

        for candidate in _candidates(tokens):
            key = fold(candidate)
            if key in self._operators:
                return OperatorScope("operator", self._operators[key], "direct", terms)
            if key in self._families:
                return OperatorScope("family", self._families[key], "direct", terms)

        return self._vote(tokens, terms)

    def _vote(self, tokens: tuple[str, ...], terms: tuple[str, ...]) -> OperatorScope:
        """Let the corpus name the family, weighting each token by its rarity."""
        if not terms or not self._total:
            return OperatorScope(None, None, "unresolved", terms)
        phrases = [
            candidate
            for candidate in _candidates(tokens)
            if "-" in candidate and all(part not in _NOISE for part in candidate.split("-"))
        ]
        scores: dict[str, float] = {}
        for family, text, words in self._documents:
            score = 0.0
            for term in terms:
                if term in words:
                    score += self._inverse_frequency(term)
            for phrase in phrases:
                if phrase.replace("-", " ") in text:
                    score += PHRASE_WEIGHT * min(
                        self._inverse_frequency(part) for part in phrase.split("-")
                    )
            if score:
                scores[family] = scores.get(family, 0.0) + score
        if not scores:
            return OperatorScope(None, None, "unresolved", terms)
        # Normalize by the square root of family size: a large family accumulates
        # incidental matches, while dividing by its full size would let a family
        # holding a single loosely-related record outrank the right one.
        ranked = sorted(
            (
                (total / math.sqrt(self._family_size[family]), family)
                for family, total in scores.items()
            ),
            reverse=True,
        )
        best, family = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
        if best < VOTE_MARGIN * runner_up:
            return OperatorScope(None, None, "ambiguous", terms)
        return OperatorScope("family", family, "voted", terms)

    def _inverse_frequency(self, term: str) -> float:
        """Rarity weight, smoothed so a common term still counts as weak evidence.

        The unsmoothed form turns negative past half the corpus, which would
        subtract from the family that actually uses the term.
        """
        return math.log1p(self._total / (1 + self._document_frequency[term]))


__all__ = ["OperatorFamilyResolver", "OperatorScope", "fold"]

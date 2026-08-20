"""Strict interchange format for provider-native Worker usage accounting."""

from __future__ import annotations

import math
import stat
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..domain.models import TokenUsage

PROVIDER_USAGE_REPORT_VERSION: Literal[2] = 2
MAX_PROVIDER_USAGE_REPORT_BYTES = 16 * 1024
type UsageUnit = Literal["provider_tokens", "credits"]
type NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
type NonNegativeNumber = Annotated[float, Field(ge=0, allow_inf_nan=False)]
type PositiveNumber = Annotated[float, Field(gt=0, allow_inf_nan=False)]
type StrictBool = Annotated[bool, Field(strict=True)]


class TokenUsageBucketsV2(BaseModel):
    """Wire representation of provider-reported token consumption."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    uncached_input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    cache_read_tokens: NonNegativeInt
    cache_write_tokens: NonNegativeInt

    @property
    def total_tokens(self) -> int:
        return (
            self.uncached_input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


class ProviderUsageReportV2(BaseModel):
    """Worker-emitted cumulative usage in the Provider's native accounting unit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2] = PROVIDER_USAGE_REPORT_VERSION
    usage_unit: UsageUnit
    budget: PositiveNumber | None
    consumed: NonNegativeNumber
    token_usage: TokenUsageBucketsV2
    credits: NonNegativeNumber | None
    budget_exhausted: StrictBool
    session_count: NonNegativeInt
    model_request_count: NonNegativeInt
    usage_complete: StrictBool

    @model_validator(mode="after")
    def _validate_derived_fields(self) -> ProviderUsageReportV2:
        if self.usage_unit == "provider_tokens":
            if self.credits is not None:
                raise ValueError("provider-token usage cannot also declare credits")
            expected = float(self.token_usage.total_tokens)
        else:
            if self.credits is None:
                raise ValueError("Qoder credit usage must declare credits")
            if self.token_usage.total_tokens != 0:
                raise ValueError("credit usage cannot also declare token consumption")
            expected = self.credits
        if not math.isclose(self.consumed, expected, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("provider usage report consumed amount is inconsistent")
        exhausted = self.budget is not None and self.consumed >= self.budget
        if self.budget_exhausted != exhausted:
            raise ValueError("provider usage report exhaustion flag is inconsistent")
        return self

    def to_domain(self) -> TokenUsage:
        """Convert validated provider-native usage into the immutable domain record."""
        return TokenUsage(
            uncached_input_tokens=self.token_usage.uncached_input_tokens,
            output_tokens=self.token_usage.output_tokens,
            cache_read_tokens=self.token_usage.cache_read_tokens,
            cache_write_tokens=self.token_usage.cache_write_tokens,
            credits=self.credits,
        )

    def require_budget(self) -> float:
        """Return a configured limit for roles whose protocol requires one."""
        if self.budget is None:
            raise ValueError("Worker provider usage report does not declare a budget")
        return self.budget

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        expected_unit: UsageUnit,
        expected_budget: float | None,
    ) -> Self:
        """Read one small regular report and verify deployment-owned accounting policy."""
        try:
            path_stat = path.lstat()
        except FileNotFoundError as error:
            raise ValueError("Worker did not produce its provider usage report") from error
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
            raise ValueError("Worker provider usage report must be a regular file")
        if path_stat.st_size > MAX_PROVIDER_USAGE_REPORT_BYTES:
            raise ValueError("Worker provider usage report exceeds the protocol size")
        report = cls.model_validate_json(path.read_bytes())
        if report.usage_unit != expected_unit:
            raise ValueError("Worker provider usage report names a different accounting unit")
        if report.budget != expected_budget:
            raise ValueError("Worker provider usage report names a different budget")
        if not report.usage_complete:
            raise ValueError(
                f"Worker model request completed without provider-reported {expected_unit}"
            )
        return report

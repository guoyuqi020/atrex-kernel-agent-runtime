"""Agent and deployment contracts reject schedules unsupported by native ABBA."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atrex_runtime.config import SameAllocationAbbaComparisonSettings
from atrex_runtime.gateway.protocol import EvaluateComparisonV2


@pytest.mark.parametrize("repeats", [2, 4, 6, 8, 10, 12, 14, 16])
def test_agent_and_controller_accept_complete_native_blocks(repeats):
    assert EvaluateComparisonV2(method="abba", repeats=repeats).repeats == repeats
    assert SameAllocationAbbaComparisonSettings(
        method="same_allocation_abba", repeats=repeats
    ).repeats == repeats


@pytest.mark.parametrize("repeats", [-2, 0, 1, 3, 15, 17, 18, 20, 21, True, False, 2.0, "2", None])
@pytest.mark.parametrize("controller", [False, True])
def test_invalid_schedules_give_actionable_validation_error(repeats, controller):
    model = SameAllocationAbbaComparisonSettings if controller else EvaluateComparisonV2
    method = "same_allocation_abba" if controller else "abba"
    with pytest.raises(ValidationError) as error:
        model.model_validate({"method": method, "repeats": repeats})
    message = error.value.errors()[0]["msg"]
    assert "2, 4, 6, 8, 10, 12, 14, 16" in message
    assert f"got {repeats!r}" in message
    assert "measurements per side; 2 means A, B, B, A" in message
    assert "not executed through Dev" in message

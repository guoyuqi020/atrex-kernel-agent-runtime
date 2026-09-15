"""Domain-specific failures."""


class InvalidTransitionError(RuntimeError):
    """Persisted state cannot perform the requested transition."""


class GatewayCapabilityPolicyChangedError(InvalidTransitionError):
    """A restarted Attempt needs a new capability recovery generation."""


class GatewayOperationsInProgressError(InvalidTransitionError):
    """Terminal handoff and active Gateway calls cannot overlap."""

    def __init__(self, operation: str, pending_operations: tuple[str, ...]) -> None:
        self.operation = operation
        self.pending_operations = pending_operations
        if operation == "attempt_report":
            detail = (
                "Attempt report was not accepted because Gateway calls are still running: "
                f"{', '.join(pending_operations)}. Wait for the existing local tool commands "
                "to finish and read their results before submitting attempt-report again. "
                "Do not launch replacement measurements or end the Session expecting a wake-up. "
                "Waiting for an existing background shell task is allowed; this is not "
                "polling or resubmitting an Agate job"
            )
        else:
            detail = (
                "Gateway call was not started because a terminal Attempt report is being "
                "submitted. Wait for that report submission to finish"
            )
        super().__init__(detail)


class InfrastructureError(RuntimeError):
    """External infrastructure failed without consuming an Agent opportunity."""

    def __init__(self, message: str = "", *, public_detail: str | None = None) -> None:
        super().__init__(message)
        # Override only where the diagnostic contains private evaluator payloads.
        self.public_detail = public_detail


class IncompleteTerminalReportError(InfrastructureError):
    """A Worker exited before Runtime accepted its required terminal report.

    The logical optimization opportunity has not produced a terminal handoff, so
    controllers may recover it with a fresh physical Session rather than consuming
    the Attempt as an ordinary negative result.
    """


class UpstreamGatewayError(InfrastructureError):
    """An upstream Gateway answered with an error available for public diagnostics.

    Only a response the Gateway actually returned carries a status. Transport
    failures stay a plain InfrastructureError because their text can embed the
    upstream URL and credentials. The HTTP boundary filters public diagnostics.
    """

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(message)


class LineageLeaseUnavailableError(RuntimeError):
    """Another trusted scheduler currently owns the requested lineage."""


class DirectionLookupError(ValueError):
    """A Direction ID did not resolve within the caller's visible history."""

    def __init__(
        self,
        requested_direction_id: str,
        suggested_direction_ids: tuple[str, ...],
        *,
        field_path: str,
    ) -> None:
        self.requested_direction_id = requested_direction_id
        self.suggested_direction_ids = suggested_direction_ids
        self.field_path = field_path
        suggestion = (
            f" Did you mean one of {list(suggested_direction_ids)}?"
            if suggested_direction_ids
            else ""
        )
        super().__init__(
            f"Direction ID {requested_direction_id!r} was not found in the current "
            f"Attempt's visible history.{suggestion} "
            "Verify the exact ID with load-direction or list-directions, correct the "
            "original request, and retry. No Journal entry was written; Runtime does "
            "not automatically replace Direction IDs"
        )


class DirectionConcurrencyError(ValueError):
    """A logical Attempt tried to explore more than one Direction concurrently."""

    def __init__(
        self,
        requested_direction_id: str,
        in_progress_direction_ids: tuple[str, ...],
    ) -> None:
        if not in_progress_direction_ids:
            raise ValueError("Direction concurrency conflict requires an active Direction")
        self.requested_direction_id = requested_direction_id
        self.in_progress_direction_ids = in_progress_direction_ids
        super().__init__(
            "Only one Direction may be in progress at a time: "
            f"requested_direction_id={requested_direction_id}; "
            f"in_progress_direction_ids={list(in_progress_direction_ids)}. "
            "The requested Direction was not started. Continue the current Direction or close it "
            "with complete, abandon, defer, or block before starting another Direction"
        )


class OptimizerSuggestionForbiddenError(ValueError):
    """An Optimizer tried to create a Bootstrap-only suggested Direction."""

    def __init__(self, field_path: str) -> None:
        self.field_path = field_path
        super().__init__(
            "Optimizer cannot use action=suggest; suggested Directions can be "
            "created only during Bootstrap"
        )


class SuggestedDirectionTransitionError(ValueError):
    """An immutable suggestion was used as a mutable Direction."""

    def __init__(self, direction_id: str, action: str, *, status: str = "suggested") -> None:
        self.direction_id = direction_id
        self.action = action
        self.status = status
        article = "An" if status[0] in "aeiou" else "A"
        super().__init__(
            f"{article} {status} Direction cannot be started, measured, or closed: "
            f"action={action}, direction_id={direction_id}. Propose your own derived "
            "Direction before exploring it"
        )


class DuplicateGatewayTaskError(ValueError):
    """An exact full-Evaluate task already completed or is currently running."""

    def __init__(self, previous_result_artifact_digest: str | None) -> None:
        self.previous_result_artifact_digest = previous_result_artifact_digest
        if previous_result_artifact_digest is None:
            detail = "the identical Gateway task is already running"
        else:
            detail = (
                "the identical Gateway task already completed; reuse Result Artifact "
                f"{previous_result_artifact_digest}"
            )
        super().__init__(detail)

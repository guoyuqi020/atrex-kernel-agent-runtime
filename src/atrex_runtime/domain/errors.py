"""Domain-specific failures."""


class InvalidTransitionError(RuntimeError):
    """Persisted state cannot perform the requested transition."""


class GatewayCapabilityPolicyChangedError(InvalidTransitionError):
    """A restarted Attempt needs a new capability recovery generation."""


class InfrastructureError(RuntimeError):
    """External infrastructure failed without consuming an Agent opportunity."""


class UpstreamGatewayError(InfrastructureError):
    """An upstream Gateway answered with an error the Agent must see verbatim.

    Only a response the Gateway actually returned carries a status. Transport
    failures stay a plain InfrastructureError because their text can embed the
    upstream URL and credentials.
    """

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(message)


class LineageLeaseUnavailableError(RuntimeError):
    """Another trusted scheduler currently owns the requested lineage."""


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

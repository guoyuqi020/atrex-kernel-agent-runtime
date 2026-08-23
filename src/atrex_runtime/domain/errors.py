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

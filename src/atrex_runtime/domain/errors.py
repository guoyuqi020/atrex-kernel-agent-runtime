"""Domain-specific failures."""


class InvalidTransitionError(RuntimeError):
    """Persisted state cannot perform the requested transition."""


class GatewayCapabilityPolicyChangedError(InvalidTransitionError):
    """A restarted Attempt needs a new capability recovery generation."""


class InfrastructureError(RuntimeError):
    """External infrastructure failed without consuming an Agent opportunity."""


class LineageLeaseUnavailableError(RuntimeError):
    """Another trusted scheduler currently owns the requested lineage."""

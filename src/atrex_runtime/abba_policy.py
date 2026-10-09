"""Native Agate ABBA limits shared by configuration, tools and execution."""

ABBA_MAX_BLOCKS = 8
ABBA_MAX_SIDE_REPEATS = 2 * ABBA_MAX_BLOCKS
ABBA_REPEATS_DESCRIPTION = (
    "Measurements per side: one of 2, 4, 6, 8, 10, 12, 14, 16. "
    "2 means one complete A, B, B, A block. Unsupported values are rejected; "
    "they do not fall back to Dev. The allocation time budget must also fit."
)


def validate_abba_repeats(value: object) -> int:
    """Reject unsupported schedules before preparing sources or submitting jobs."""
    if type(value) is not int or value < 2 or value > ABBA_MAX_SIDE_REPEATS or value % 2:
        raise ValueError(
            "ABBA repeats must be one of 2, 4, 6, 8, 10, 12, 14, 16 "
            f"(measurements per side; 2 means A, B, B, A); got {value!r}. "
            "Unsupported repeats are rejected, not executed through Dev."
        )
    return value

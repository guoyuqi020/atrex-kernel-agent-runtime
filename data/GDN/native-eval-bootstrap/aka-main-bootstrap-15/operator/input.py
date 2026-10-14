import torch
import torch.nn.functional as F


def _make_inputs(
    total_tokens: int,
    num_q_heads: int,
    num_v_heads: int,
    sequence_lengths_values: list[int],
    initial_state_mode: str,
    head_dim: int,
    qkv_dtype: str,
    gate_dtype: str,
    beta_dtype: str,
    state_dtype: str,
) -> dict[str, torch.Tensor]:
    if head_dim != 128:
        raise ValueError("SM103 M64 requires head_dim=128")
    if qkv_dtype != "fp16":
        raise ValueError("SM103 M64 requires fp16 q/k/v/output")
    if gate_dtype != "fp32" or beta_dtype != "fp32" or state_dtype != "fp32":
        raise ValueError("SM103 M64 requires fp32 gate/beta/state")
    if len(sequence_lengths_values) not in (1, 2):
        raise ValueError("SM103 M64 requires one or two packed sequences")
    if any(length <= 0 for length in sequence_lengths_values):
        raise ValueError("packed sequence lengths must be positive")
    if sum(sequence_lengths_values) != total_tokens:
        raise ValueError("sequence_lengths_values must sum to total_tokens")
    if num_v_heads % num_q_heads != 0:
        raise ValueError("num_v_heads must be divisible by num_q_heads")
    if initial_state_mode not in {"zero", "nonzero"}:
        raise ValueError("initial_state_mode must be zero or nonzero")

    device = "cuda"
    q = (
        F.normalize(
            torch.randn(
                total_tokens,
                num_q_heads,
                head_dim,
                dtype=torch.float32,
                device=device,
            ),
            dim=-1,
        )
        .to(torch.float16)
        .contiguous()
    )
    k = (
        F.normalize(
            torch.randn(
                total_tokens,
                num_q_heads,
                head_dim,
                dtype=torch.float32,
                device=device,
            ),
            dim=-1,
        )
        .to(torch.float16)
        .contiguous()
    )
    v = torch.randn(
        total_tokens,
        num_v_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
    ).contiguous()
    g = (
        torch.rand(
            total_tokens,
            num_v_heads,
            dtype=torch.float32,
            device=device,
        )
        .clamp_(min=2.0**-12, max=1.0)
        .contiguous()
    )
    beta = torch.rand(
        total_tokens,
        num_v_heads,
        dtype=torch.float32,
        device=device,
    ).contiguous()

    num_sequences = len(sequence_lengths_values)
    initial_state = torch.zeros(
        num_sequences,
        num_v_heads,
        head_dim,
        head_dim,
        dtype=torch.float32,
        device=device,
    )
    if initial_state_mode == "nonzero":
        initial_state.normal_(mean=0.0, std=0.1)

    endpoints = [0]
    for length in sequence_lengths_values:
        endpoints.append(endpoints[-1] + length)

    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "initial_state": initial_state.contiguous(),
        "cu_seqlens": torch.tensor(endpoints, dtype=torch.int32, device=device),
        "output": torch.empty_like(v),
        "output_state": torch.empty_like(initial_state),
    }

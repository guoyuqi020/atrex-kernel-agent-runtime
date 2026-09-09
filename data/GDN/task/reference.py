import torch
import torch.nn as nn


class Model(nn.Module):
    """Prepared-input GDN prefill recurrence used by the SM103 M64 task."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        output: torch.Tensor,
        output_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        total_tokens, num_q_heads, head_dim = q.shape
        num_v_heads = v.shape[1]
        repeats = num_v_heads // num_q_heads
        scale = head_dim**-0.5

        q_float = q.float().repeat_interleave(repeats, dim=1) * scale
        k_float = k.float().repeat_interleave(repeats, dim=1)
        v_float = v.float()
        bounds = [int(value) for value in cu_seqlens.tolist()]

        for request_index, (start, end) in enumerate(zip(bounds, bounds[1:])):
            state = initial_state[request_index].float().clone()
            for token_index in range(start, end):
                state = state * g[token_index].view(num_v_heads, 1, 1)
                key = k_float[token_index]
                reconstructed = (state * key.unsqueeze(1)).sum(dim=-1)
                delta = (v_float[token_index] - reconstructed) * beta[token_index].unsqueeze(-1)
                state = state + delta.unsqueeze(-1) * key.unsqueeze(1)
                output[token_index] = (
                    (state * q_float[token_index].unsqueeze(1)).sum(dim=-1).to(output.dtype)
                )
            output_state[request_index] = state.to(output_state.dtype)

        if output.shape != (total_tokens, num_v_heads, head_dim):
            raise ValueError("output shape does not match the GDN result")
        return {"output": output, "final_state": output_state}

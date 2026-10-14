"""The same FP8 input regime as the original upstream P128 smoke."""
import torch


def _make_inputs():
    device = torch.device("cuda", 0)
    return {
        "q": (torch.randn(128, 16, 256, dtype=torch.bfloat16, device=device) * 0.1).to(
            torch.float8_e4m3fn),
        "k": (torch.randn(1, 128, 1, 256, dtype=torch.bfloat16, device=device) * 0.1).to(
            torch.float8_e4m3fn),
        "v": (torch.randn(1, 128, 1, 256, dtype=torch.bfloat16, device=device) * 0.1).to(
            torch.float8_e4m3fn),
        "cu_q": torch.tensor([0, 128], dtype=torch.int32, device=device),
        "page_table": torch.tensor([[0]], dtype=torch.int32, device=device),
    }

"""Same stored inputs, independent FP32 GDN recurrence and gated RMSNorm."""
import torch
from torch.nn import functional as F

class Model(torch.nn.Module):

    def __init__(self, fault=None):
        super().__init__()
        self.fault = fault

    def forward(self, mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state, output_gate, norm_weight, out=None, scale=128 ** (-0.5), norm_eps=1e-05, output_gate_activation='silu'):
        hv = state.shape[1]
        h = (mixed_qkv.shape[1] - hv * 128) // 256
        t = len(mixed_qkv)
        q, k, v = mixed_qkv.float().split((h * 128, h * 128, hv * 128), -1)
        q = q.view(t, h, 128)
        k = k.view(t, h, 128)
        v = v.view(t, hv, 128)
        eps = 0.0 if self.fault == 'no_qk_epsilon' else 1e-06
        if self.fault != 'no_qk_norm':
            q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + eps)
            k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + eps)
        q = q * scale
        if self.fault == 'wrong_gqa':
            q = q.repeat(1, hv // h, 1)
            k = k.repeat(1, hv // h, 1)
        else:
            q = q.repeat_interleave(hv // h, 1)
            k = k.repeat_interleave(hv // h, 1)
        decay = torch.exp(-A_log.float().exp() * F.softplus(a.float() + dt_bias.float()))
        beta = torch.sigmoid(b.float())
        gate = output_gate.float()
        gate = torch.sigmoid(gate) if output_gate_activation == 'sigmoid' else F.silu(gate)
        if self.fault == 'no_decay':
            decay = torch.ones_like(decay)
        if self.fault == 'no_beta':
            beta = torch.ones_like(beta)
        if self.fault == 'no_gate':
            gate = torch.ones_like(gate)
        if self.fault == 'wrong_gate':
            gate = F.silu(output_gate.float())
        if out is None:
            out = torch.empty_like(output_gate)
        out.zero_()
        cu = cu_seqlens.tolist()
        slots = state_indices.tolist()
        accepted = num_accepted_tokens.tolist()
        if self.fault == 'ignored_accepted':
            accepted = [1] * len(accepted)
        active = [i for i, (s, z) in enumerate(zip(slots, accepted)) if 0 < cu[i + 1] - cu[i] <= 8 and 0 < z <= len(s) and (s[z - 1] > 0)]
        if not active:
            return out
        src = torch.tensor([slots[i][accepted[i] - 1] for i in active], device=state.device)
        carried = state[src].float()
        for step in range(max((cu[i + 1] - cu[i] for i in active))):
            rows = [r for r, i in enumerate(active) if step < cu[i + 1] - cu[i]]
            tokens = torch.tensor([cu[active[r]] + step for r in rows], device=state.device)
            rr = torch.tensor(rows, device=state.device)
            hh = carried[rr] * decay[tokens, :, None, None]
            pred = (hh * k[tokens, :, None, :]).sum(-1)
            delta = (v[tokens] - pred) * beta[tokens, :, None]
            hh = hh + delta[..., None] * k[tokens, :, None, :]
            carried[rr] = hh
            raw = (hh * q[tokens, :, None, :]).sum(-1)
            normalized = raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + norm_eps)
            if self.fault == 'no_norm':
                normalized = raw
            out[tokens] = (normalized * norm_weight.float() * gate[tokens]).to(out.dtype)
            valid = [j for j, r in enumerate(rows) if slots[active[r]][step] > 0]
            dst = torch.tensor([slots[active[rows[j]]][step] for j in valid], device=state.device, dtype=torch.long)
            if valid:
                state[dst] = hh[valid].to(state.dtype)
        return out

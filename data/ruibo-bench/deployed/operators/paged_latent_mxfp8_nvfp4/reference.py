"""Independent byte decoding followed by FP32 attention, not native rounding."""
import torch

def read_rows(cache, slots):
    pages, page, _, record = cache.shape
    fp4 = record == 288
    width, group = (256, 16) if fp4 else (512, 32)
    raw = cache.reshape(pages, -1)
    slots = slots.long()
    p, r = (slots // page, slots % page)
    data = raw[p[:, None], r[:, None] * width + torch.arange(width, device=cache.device)]
    scale = raw[p[:, None], page * width + r[:, None] * (512 // group) + torch.arange(512 // group, device=cache.device)]
    if fp4:
        codes = torch.stack((data & 15, data >> 4), -1).reshape(-1, 512)
        lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=cache.device)
        value = lut[(codes & 7).long()] * torch.where(codes & 8 != 0, -1.0, 1.0)
        scale = scale.contiguous().view(torch.float8_e4m3fn).float()
    else:
        value = data.contiguous().view(torch.float8_e4m3fn).float()
        scale = torch.exp2(scale.float() - 127)
    return (value.reshape(-1, 512 // group, group) * scale[..., None]).reshape(-1, 512)

def evaluate(q, k_cache, indices, extra_k_cache, extra_indices, sm_scale, attn_sink, out):
    b, s, h, _ = q.shape
    result = torch.empty_like(out)
    lse = torch.empty(b, s, h, dtype=torch.float32, device=q.device).transpose(1, 2)
    for r in range(b):
        for j in range(s):
            selected = []
            for kv, ids in ((k_cache, indices), (extra_k_cache, extra_indices)):
                if kv is None:
                    continue
                ix = ids[r, j]
                ix = ix[ix >= 0]
                if len(ix):
                    selected.append(read_rows(kv, ix))
            if not selected:
                result[r, j] = 0
                lse[r, :, j] = torch.inf
                continue
            k = torch.cat(selected)
            logits = q[r, j].float() @ k.T * sm_scale
            sums = torch.logsumexp(logits, -1)
            lse[r, :, j] = sums
            result[r, j] = (torch.exp(logits - torch.logaddexp(sums, attn_sink)[:, None]) @ k).to(out.dtype)
    return [result, lse]

class Model(torch.nn.Module):

    def forward(self, q, k_cache, indices, extra_k_cache, extra_indices, sm_scale, attn_sink, out):
        torch.backends.cuda.matmul.allow_tf32 = False
        values = evaluate(q, k_cache, indices, extra_k_cache, extra_indices, sm_scale, attn_sink, out)
        out.copy_(values[0])
        return (out, values[1])

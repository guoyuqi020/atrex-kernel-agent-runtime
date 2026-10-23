import torch

def decode4(q, sf):
    m, k = (q.shape[0], q.shape[1] * 2)
    sf = sf.reshape((m + 127) // 128, (k + 63) // 64, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape((m + 127) // 128 * 128, -1)[:m, :k // 16].contiguous().view(torch.float8_e4m3fn)
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device=q.device)
    return lut[torch.stack((q & 15, q >> 4), -1).long()].reshape(m, k) * sf.float().repeat_interleave(16, -1)

def expected(p):
    y = torch.zeros(p['out'].shape, dtype=torch.float32, device=p['a'].device)
    ends = p['m_indptr'].tolist()
    for e, (lo, hi) in enumerate(zip(ends, ends[1:])):
        if hi == lo:
            continue
        if p['a'].dtype == torch.uint8:
            sflo = (lo + e * 127) // 128 * 128
            sfhi = sflo + (hi - lo + 127) // 128 * 128
            a = decode4(p['a'][lo:hi], p['a_scale'][sflo:sfhi])
            b = decode4(p['b'][e], p['b_scale'][e])
        else:
            a = p['a'][lo:hi].float() * p['a_scale'][:, lo:hi].T.repeat_interleave(128, -1)
            b = p['b'][e].float() * p['b_scale'][e].T.repeat_interleave(128, 0).repeat_interleave(128, -1)
        for off in range(0, hi - lo, 64):
            y[lo + off:lo + min(off + 64, hi - lo)] = torch.einsum('mk,nk->mn', a[off:off + 64], b)
        if p.get('alpha') is not None:
            y[lo:hi] *= p['alpha'][e]
    return y.to(p['out'].dtype)

class Model(torch.nn.Module):

    def forward(self, a, b, a_scale, b_scale, m_indptr, scale_granularity_mnk=(1, 128, 128), scale_major_mode='MN', mma_sm=1, out=None, out_dtype=None, backend='trtllm', segment_alignment=1):
        torch.backends.cuda.matmul.allow_tf32 = False
        p = {'a': a, 'b': b, 'a_scale': a_scale, 'b_scale': b_scale, 'm_indptr': m_indptr, 'scale_granularity_mnk': scale_granularity_mnk, 'scale_major_mode': scale_major_mode, 'mma_sm': mma_sm, 'out': out, 'out_dtype': out_dtype, 'backend': backend, 'segment_alignment': segment_alignment}
        result = expected(p)
        out.copy_(result)
        return out

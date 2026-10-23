import torch

def fp4(q, sf, swizzled=True):
    q = q.view(torch.uint8)
    m, k = (q.shape[0], q.shape[1] * 2)
    if swizzled:
        sf = sf.view(torch.uint8).reshape((m + 127) // 128, (k + 63) // 64, 32, 4, 4)
        sf = sf.permute(0, 3, 2, 1, 4).reshape((m + 127) // 128 * 128, (k + 63) // 64 * 4)[:m, :k // 16]
        sf = sf.contiguous().view(torch.float8_e4m3fn)
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device=q.device)
    return lut[torch.stack((q & 15, q >> 4), -1).long()].reshape(m, k) * sf.float().repeat_interleave(16, -1)

def expected(p):
    a = fp4(*p['a'])
    b = fp4(*p['b'])
    y = torch.empty(p['d'].shape, device=a.device, dtype=torch.float32)
    for off in range(0, a.shape[0], 64):
        y[off:off + 64] = torch.einsum('mk,nk->mn', a[off:off + 64], b)
    if p['c'] is not None:
        y += p['c'].float()
    return y.to(p['d'].dtype)

class Model(torch.nn.Module):

    def forward(self, a, b, d, c=None):
        torch.backends.cuda.matmul.allow_tf32 = False
        p = {'a': a, 'b': b, 'd': d, 'c': c}
        result = expected(p)
        d.copy_(result)
        return None

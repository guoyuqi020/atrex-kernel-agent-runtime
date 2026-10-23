import math
import torch

def g(shape, device, std=1.0, role='projection'):
    n = torch.randn(shape, device=device)
    b = torch.linspace(0.65, 1.35, shape[-1], device=device)
    n.mul_(b)
    if role == 'fc2':
        c = torch.randn_like(n)
        m = 0.25 * c + math.sqrt(1 - 0.25 ** 2) * n
        n = torch.nn.functional.silu(c) * m
    elif role == 'weight':
        n.mul_(torch.linspace(0.7, 1.3, shape[-2], device=device)[:, None])
    elif role != 'projection':
        raise ValueError(role)
    n.mul_(std)
    k = n.reshape(-1)[:1 << 18].float()
    return n
'Canonical stored-precision inputs; no dependency on a serving framework.'
import math
import torch
h = g
i = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

def l(shape, dtype, device):
    b = math.prod(shape)
    a = torch.full((b + 128,), -17, dtype=dtype, device=device)
    return a[64:-64].view(shape)

def o(value, block, global_scale=1.0, scale_rows=1):
    m = value.shape[-1]
    assert m % block == 0
    j = value.float().reshape(*value.shape[:-1], m // block, block)
    v = j.abs().amax(-1).clamp_min(2.0 ** (-20))
    if scale_rows != 1:
        assert value.shape[-2] % scale_rows == 0
        v = v.reshape(*v.shape[:-2], value.shape[-2] // scale_rows, scale_rows, v.shape[-1]).amax(-2)
    a = 6.0
    w = v / a / global_scale
    x = w.clamp(2.0 ** (-9), 448).to(torch.float8_e4m3fn)
    w = x.float()
    d = w.repeat_interleave(scale_rows, -2)
    s = (j / (d.unsqueeze(-1) * global_scale)).reshape_as(value)
    n = s.abs()
    b = torch.zeros_like(n, dtype=torch.uint8)
    for k in range(7):
        p = (i[k] + i[k + 1]) / 2
        c = n > p if k % 2 == 0 else n >= p
        b += c.to(torch.uint8)
    b |= torch.signbit(s).to(torch.uint8) << 3
    u = b[..., 0::2] | b[..., 1::2] << 4
    return (u, x)

def q(shape, block, device, std=1.0, zero=False, global_scale=1.0, scale_rows=1, profile=None, input_role='projection'):
    a = h(shape, device, std, role='weight' if std != 1.0 else input_role)
    if profile == 'channel_outliers':
        a[..., ::64] *= 8.0
    if zero:
        a.zero_()
    return o(a, block, global_scale, scale_rows)

def r(sf):
    b, a = sf.shape
    c = torch.zeros(((b + 127) // 128 * 128, (a + 3) // 4 * 4), dtype=torch.uint8, device=sf.device)
    c[:b, :a] = sf.view(torch.uint8)
    return c.reshape(c.shape[0] // 128, 4, 32, c.shape[1] // 4, 4).permute(0, 3, 2, 1, 4).contiguous()

def t(s, seed=0, device='cuda'):
    j = 'fc2' if (s['n'], s['k']) in ((4096, 12288), (7168, 2048), (4096, 1536)) else 'projection'
    torch.manual_seed(seed)
    m, n, k = (s['m'], s['n'], s['k'])
    a, p = q((m, k), 16, device, zero=s.get('distribution') == 'zero', profile=s.get('distribution'), input_role=j)
    b, u = q((n, k), 16, device, std=k ** (-0.5), input_role=j)
    d = l((m, n), torch.bfloat16, device)
    c = torch.randn((m, n), device=device).to(d.dtype) if s.get('accumulate') else None
    if s.get('c_alias_d'):
        d.copy_(torch.randn((m, n), device=device))
        c = d
    return dict(a=(a.view(torch.int8), r(p)), b=(b.view(torch.int8), r(u)), d=d, c=c)

def _make_inputs(m, n, k, accumulate=None, c_alias_d=None, distribution=None, device='cuda'):
    b = {'m': m, 'n': n, 'k': k, **{k: c for k, c in {'accumulate': accumulate, 'c_alias_d': c_alias_d, 'distribution': distribution}.items() if c is not None}}
    a = t(b, seed=torch.initial_seed(), device=device)
    return {'a': a['a'], 'b': a['b'], 'd': a['d'], 'c': a['c']}

def writable_views(x):
    b = {a: x[a] for a in ['d']}
    if torch.is_tensor(x.get('c')):
        b['c'] = x['c']
    return b

def observable_inputs(x):
    b = writable_views(x)
    return {a: b[a] for a in ['d']}

def replay_tensors(x):
    return writable_views(x)

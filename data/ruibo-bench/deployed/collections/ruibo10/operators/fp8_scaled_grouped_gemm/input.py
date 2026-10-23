import math
import torch

def f(shape, device, std=1.0, role='projection'):
    p = torch.randn(shape, device=device)
    b = torch.linspace(0.65, 1.35, shape[-1], device=device)
    p.mul_(b)
    if role == 'fc2':
        e = torch.randn_like(p)
        o = 0.25 * e + math.sqrt(1 - 0.25 ** 2) * p
        p = torch.nn.functional.silu(e) * o
    elif role == 'weight':
        p.mul_(torch.linspace(0.7, 1.3, shape[-2], device=device)[:, None])
    elif role != 'projection':
        raise ValueError(role)
    p.mul_(std)
    n = p.reshape(-1)[:1 << 18].float()
    return p
'Canonical stored-precision inputs; no dependency on a serving framework.'
import math
import torch
g = f

def h(shape, dtype, device):
    b = math.prod(shape)
    a = torch.full((b + 128,), -17, dtype=dtype, device=device)
    return a[64:-64].view(shape)

def i(value, block, global_scale=1.0, scale_rows=1):
    k = value.shape[-1]
    assert k % block == 0
    e = value.float().reshape(*value.shape[:-1], k // block, block)
    o = e.abs().amax(-1).clamp_min(2.0 ** (-20))
    if scale_rows != 1:
        assert value.shape[-2] % scale_rows == 0
        o = o.reshape(*o.shape[:-2], value.shape[-2] // scale_rows, scale_rows, o.shape[-1]).amax(-2)
    a = 448.0
    p = o / a / global_scale
    q = p
    b = p.repeat_interleave(scale_rows, -2)
    m = (e / (b.unsqueeze(-1) * global_scale)).reshape_as(value)
    n = m.clamp(-448, 448).to(torch.float8_e4m3fn)
    return (n, q)

def j(shape, block, device, std=1.0, zero=False, global_scale=1.0, scale_rows=1, input_role='projection'):
    if len(shape) == 3:
        b = [j(shape[1:], block, device, std, zero, global_scale, scale_rows, input_role=input_role) for a in range(shape[0])]
        e = torch.stack([n[0] for n in b])
        k = None if b[0][1] is None else torch.stack([n[1] for n in b])
        return (e, k)
    m = g(shape, device, std, role='weight' if std != 1.0 else input_role)
    if zero:
        m.zero_()
    return i(m, block, global_scale, scale_rows)

def l(s, seed=0, device='cuda'):
    p = 'fc2' if (s['n'], s['k']) in ((4096, 12288), (7168, 2048), (4096, 1536)) else 'projection'
    torch.manual_seed(seed)
    m = s['group_sizes']
    r = sum(m)
    n = len(m)
    t, q = (s['n'], s['k'])
    u = [0]
    for k in m:
        u.append(u[-1] + k)
    o = torch.tensor(u, dtype=torch.int32, device=device)
    e = 128
    a, w = j((r, q), e, device, zero=s.get('distribution') == 'zero', input_role=p)
    b, x = j((n, t, q), e, device, std=q ** (-0.5), scale_rows=128, input_role=p)
    v = h((r, t), torch.bfloat16, device)
    return dict(a=a, b=b, a_scale=w.T.contiguous(), b_scale=x.transpose(-1, -2).contiguous(), m_indptr=o, scale_granularity_mnk=(1, 128, 128), scale_major_mode='MN', mma_sm=1, out=v, out_dtype=torch.bfloat16, backend='trtllm', segment_alignment=1)

def _make_inputs(m, n, k, groups, b_scale_rows=None, group_sizes=None, logical_group_sizes=None, mode=None, device='cuda'):
    b = {'m': m, 'n': n, 'k': k, **{k: e for k, e in {'b_scale_rows': b_scale_rows, 'group_sizes': group_sizes, 'groups': groups, 'logical_group_sizes': logical_group_sizes, 'mode': mode}.items() if e is not None}}
    a = l(b, seed=torch.initial_seed(), device=device)
    return {'a': a['a'], 'b': a['b'], 'a_scale': a['a_scale'], 'b_scale': a['b_scale'], 'm_indptr': a['m_indptr'], 'scale_granularity_mnk': a['scale_granularity_mnk'], 'scale_major_mode': a['scale_major_mode'], 'mma_sm': a['mma_sm'], 'out': a['out'], 'out_dtype': a['out_dtype'], 'backend': a['backend'], 'segment_alignment': a['segment_alignment']}

def writable_views(x):
    b = {a: x[a] for a in ['out']}
    return b

def observable_inputs(x):
    b = writable_views(x)
    return {a: b[a] for a in ['out']}

def replay_tensors(x):
    return writable_views(x)

import torch

def l(count, layout, device='cpu'):
    if layout == 'contiguous':
        return torch.arange(count, device=device)
    if layout == 'scattered':
        return torch.randperm(count, device=device)
    raise ValueError(layout)

def m(visible, budget, pattern, scores=None, device='cpu'):
    a = min(visible, budget)
    if a == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if pattern == 'score':
        assert scores is not None and scores.numel() >= visible
        return scores[:visible].topk(a).indices.to(device)
    if pattern == 'local':
        return torch.arange(visible - a, visible, device=device)
    if pattern == 'shared_prefix':
        return torch.arange(a, device=device)
    if pattern == 'strided':
        return torch.linspace(0, visible - 1, a, device=device).round().long()
    if pattern == 'random':
        return torch.randperm(visible, device=device)[:a]
    raise ValueError(pattern)

def o(indices, page_size, *, name='main', limit=None):
    h = indices.reshape(-1, indices.shape[-1])
    b = []
    e = None
    for g in h[:16]:
        a = g[g >= 0]
        if limit is not None:
            a = a[a < limit]
        i = set(a.cpu().tolist())
        d = {j // page_size for j in i}
        c = len(i & e) / max(1, len(i | e)) if e is not None else None
        b.append(dict(selected=len(i), unique_pages=len(d), selected_per_page=len(i) / max(1, len(d)), adjacent_query_jaccard=c))
        e = i
import torch

def t(x):
    a = torch.full((x.numel() + 256,), 73, dtype=x.dtype, device=x.device)
    b = a[128:-128].view(x.shape)
    b.copy_(x)
    return b

def u(x, mixed):
    n, k, a = x.shape
    q = 16 if mixed else 32
    g = x.float().reshape(n, k, 512 // q, q)
    b = g.abs().amax(-1)
    if mixed:
        p = (b / 6).clamp(2.0 ** (-9), 448).to(torch.float8_e4m3fn)
        r = g / p.float()[..., None]
        i = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=x.device)
        j = (i[1:] + i[:-1]) * 0.5
        c = torch.zeros_like(r, dtype=torch.uint8)
        for h in range(7):
            c += ((r.abs() > j[h]) | (r.abs() == j[h]) & (h % 2 == 1)).to(torch.uint8)
        c |= torch.signbit(r).to(torch.uint8) << 3
        c = c.reshape(n, k, 512)
        d = c[..., ::2] | c[..., 1::2] << 4
        p = p.view(torch.uint8)
    else:
        e = torch.ceil(torch.log2(b.clamp_min(0.0001) / 448))
        p = (e + 127).to(torch.uint8)
        d = (g / torch.exp2(e)[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    return torch.cat((d.reshape(n, -1), p.reshape(n, -1)), -1).reshape(n, k, 1, 288 if mixed else 528)
'Per-query causal sparse slots over the original page-SoA low-precision caches.'
import torch
w = l
aa = m
ab = o
ac = u
ad = t

def ae(page, compression, window, budget, fp4, *, lengths, query_len, physical_layout, device, centers, amp, q, index_pattern, b):
    ah = [max(0, (n - query_len + 1) // compression - window) // page if window else 0 for n in lengths]
    g = [(n // compression + page - 1) // page for n in lengths]
    d = [e - ag for ag, e in zip(ah, g)]
    r = sum(d) + 2
    v = w(r, physical_layout)
    k = torch.empty(r, page, 1, 288 if fp4 else 528, dtype=torch.uint8, device=device)
    h = torch.empty((r, page, 8), device=device)
    for y in range(0, r, 64):
        ai = (0.8 * torch.randn(min(64, r - y), page, 512, device=device) * amp + 0.6 * centers[torch.randint(4, (min(64, r - y), page), device=device)] * (amp / 0.35)).to(torch.bfloat16)
        h[y:y + len(ai)] = ai[:, :, :8].float()
        k[y:y + len(ai)] = ac(ai, fp4)
    i = torch.full((b, query_len, budget), -1, dtype=torch.int32, device=device)
    p = 0
    for x, (n, ag, c) in enumerate(zip(lengths, ah, d)):
        s = v[p:p + c]
        p += c
        for j in range(query_len):
            aj = (n - query_len + j + 1) // compression
            z = None if window else h[s.to(device)].reshape(-1, 8) @ q[x, j, :, :8].float().mean(0)
            a = torch.arange(max(0, aj - window), aj) if window else aa(aj, budget, index_pattern, z)
            if len(a):
                af = s[a // page - ag] * page + a % page
                i[x, j, :len(af)] = af.to(device=device, dtype=torch.int32)
    ab(i, page, name='swa' if window else 'compressed')
    return (k, i)

def _make_inputs(lengths, tp, ratio, query_len, domain='normal', mixed=True, physical_heads=64, device='cuda', physical_layout='scattered', index_pattern='score'):
    assert mixed is True
    assert query_len in (1, 3) and physical_heads in (64, 128) and (tp in (1, 2, 4, 8))
    assert ratio in (0, 1, 2) and all((query_len <= k <= 1048576 for k in lengths))
    b = len(lengths)
    g = physical_heads
    a = 0.0 if domain == 'zero' else 0.35 * 2 ** (-12) if domain == 'tiny' else 0.35
    c = torch.randn((4, 512), device=device) * 0.35
    p = 0.8 * torch.randn(b, query_len, g, 512, device=device) * 0.35 + 0.6 * c[torch.randint(4, (b, query_len, g), device=device)]
    if domain == 'peaked':
        p *= 16
    p = p.to(torch.bfloat16)
    if domain == 'tiny':
        p *= 2 ** (-8)
    j, h = ae(32, 1, 128, 128, False, lengths=lengths, query_len=query_len, physical_layout=physical_layout, device=device, centers=c, amp=a, q=p, index_pattern=index_pattern, b=b)
    e, d = ae(128 // ratio, ratio, 0, 512, mixed, lengths=lengths, query_len=query_len, physical_layout=physical_layout, device=device, centers=c, amp=a, q=p, index_pattern=index_pattern, b=b) if ratio else (None, None)
    q = torch.randn(g, device=device)
    if domain == 'sink':
        q[:] = 10.0
        q[0] = torch.inf
        q[1] = -torch.inf
    s = dict(q=p, k_cache=j, indices=h, extra_k_cache=e, extra_indices=d, sm_scale=512 ** (-0.5), attn_sink=q, out=torch.empty_like(p))
    n = {i: ad(r) if isinstance(r, torch.Tensor) else r for i, r in s.items()}
    return {'q': n['q'], 'k_cache': n['k_cache'], 'indices': n['indices'], 'extra_k_cache': n['extra_k_cache'], 'extra_indices': n['extra_indices'], 'sm_scale': n['sm_scale'], 'attn_sink': n['attn_sink'], 'out': n['out']}

def writable_views(x):
    b = {a: x[a] for a in ['out']}
    return b

def observable_inputs(x):
    b = writable_views(x)
    return {a: b[a] for a in ['out']}

def replay_tensors(x):
    return writable_views(x)

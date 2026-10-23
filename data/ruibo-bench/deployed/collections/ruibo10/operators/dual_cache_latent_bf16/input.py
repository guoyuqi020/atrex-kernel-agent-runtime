import torch

def b(count, layout, device='cpu'):
    if layout == 'contiguous':
        return torch.arange(count, device=device)
    if layout == 'scattered':
        return torch.randperm(count, device=device)
    raise ValueError(layout)

def c(visible, budget, pattern, scores=None, device='cpu'):
    j = min(visible, budget)
    if j == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if pattern == 'score':
        assert scores is not None and scores.numel() >= visible
        return scores[:visible].topk(j).indices.to(device)
    if pattern == 'local':
        return torch.arange(visible - j, visible, device=device)
    if pattern == 'shared_prefix':
        return torch.arange(j, device=device)
    if pattern == 'strided':
        return torch.linspace(0, visible - 1, j, device=device).round().long()
    if pattern == 'random':
        return torch.randperm(visible, device=device)[:j]
    raise ValueError(pattern)

def d(indices, page_size, *, name='main', limit=None):
    q = indices.reshape(-1, indices.shape[-1])
    k = []
    o = None
    for p in q[:16]:
        j = p[p >= 0]
        if limit is not None:
            j = j[j < limit]
        r = set(j.cpu().tolist())
        n = {s // page_size for s in r}
        m = len(r & o) / max(1, len(r | o)) if o is not None else None
        k.append(dict(selected=len(r), unique_pages=len(n), selected_per_page=len(r) / max(1, len(n)), adjacent_query_jaccard=m))
        o = r
import math
import torch

def f(nq, nk, heads, kv_heads, dim, value_dim, device, profile='normal', scale=None):
    assert heads % kv_heads == 0
    w = torch.arange(max(dim, value_dim), device=device).float()
    k = torch.sin((torch.arange(1, 9, device=device).float()[:, None] * w[None, :] + 0.3) / 17.0)
    k = k / k.square().mean(-1, keepdim=True).sqrt() / math.sqrt(8)
    u = torch.randn((kv_heads, 4, 8), device=device)
    m = u @ k[:, :dim]
    ag = u @ torch.cos((torch.arange(1, 9, device=device).float()[:, None] * w[None, :value_dim] + 0.7) / 19.0) / 2.0
    r = torch.empty((nk, kv_heads, dim), device=device)
    af = torch.empty((nk, kv_heads, value_dim), device=device)
    s = torch.linspace(0.7, 1.3, kv_heads, device=device)[None, :, None]
    n = torch.linspace(0.7, 1.3, dim, device=device)
    p = torch.arange(kv_heads, device=device)[None, :]
    for ad in range(0, nk, 1024):
        o = min(1024, nk - ad)
        ae = torch.randint(4, (o, kv_heads), device=device)
        r[ad:ad + o] = (0.8 * torch.randn((o, kv_heads, dim), device=device) + 0.6 * m[p, ae]) * s * n
        af[ad:ad + o] = (0.8 * torch.randn((o, kv_heads, value_dim), device=device) + 0.6 * ag[p, ae]) * s
    y = torch.empty((nq, heads, dim), device=device)
    z = m.repeat_interleave(heads // kv_heads, 0)
    ab = torch.arange(heads, device=device)[None, :]
    aa = torch.linspace(0.65, 1.35, heads, device=device)[None, :, None]
    for ad in range(0, nq, 1024):
        o = min(1024, nq - ad)
        ae = torch.randint(4, (o, heads), device=device)
        y[ad:ad + o] = (0.8 * torch.randn((o, heads, dim), device=device) + 0.6 * z[ab, ae]) * aa * n
    if dim == 576:
        y[..., 512:] *= 0.5
        r[..., 512:] *= 0.5
    ac = y[:min(nq, 8), ::max(1, heads // 4)].float()
    q = torch.arange(heads, device=device)[::max(1, heads // 4)] // (heads // kv_heads)
    t = r[:min(nk, 256), q].float()
    return (y, r, af)
'V4 Flash/Pro sparse MLA inputs: C4 top-k, C128 all compressed, SWA128.\n\nModel-derived synthetic cache allocation and legal selection, not traffic or\nindexer validation. Cache writes/selection/RoPE are outside the timed callable.\n'
import torch
g = b
h = c
i = d
l = f

def _make_inputs(query_lens, seq_lens_values, num_heads, compress_ratio, index_topk, page_size, distribution='normal', device='cuda', index_pattern='random', physical_layout='scattered'):
    assert compress_ratio in (0, 1, 2, 4, 128) and num_heads in (8, 16, 32, 64, 128)
    assert index_topk in (512, 1024)
    assert len(query_lens) == len(seq_lens_values) and query_lens
    assert all((0 < ai <= al <= 1048576 for ai, al in zip(query_lens, seq_lens_values)))
    assert distribution in ('normal', 'tiny', 'peaked', 'strong_sink')
    assert page_size in (128, 256) and page_size % max(1, compress_ratio) == 0
    assert index_pattern in ('random', 'local', 'strided', 'shared_prefix', 'score')
    s, af = (512, page_size)
    p = af // max(1, compress_ratio)
    r = [0]
    for z in query_lens:
        r.append(r[-1] + z)
    aq = [list(range(max(0, al - ai - 127) // af, (al + af - 1) // af)) for ai, al in zip(query_lens, seq_lens_values)]
    ah = g(sum(map(len, aq)), physical_layout).tolist()
    ar, au = ([], 0)
    for ag in aq:
        ar.append(dict(zip(ag, ah[au:au + len(ag)])))
        au += len(ag)
    n = [(al // compress_ratio + p - 1) // p if compress_ratio else 0 for al in seq_lens_values]
    ah = g(sum(n), physical_layout).tolist()
    o, au = ([], 0)
    for q in n:
        o.append(ah[au:au + q])
        au += q
    ad = sum(map(len, aq)) * af
    ab = max(1, sum(n)) * p
    ai, y, j = l(r[-1], ad + ab, num_heads, 1, s, 0, device)
    ai = ai * 0.5
    ap = y[:ad].reshape(-1, af, s) * 0.25 - 0.0625
    v = y[ad:].reshape(-1, p, s) * 0.25 + 0.0625
    an = v[:, :, :8].float().clone()
    del y
    if distribution == 'tiny':
        ap *= 2.0 ** (-10)
        v *= 2.0 ** (-10)
    elif distribution == 'peaked':
        ai *= 6
        ap *= 4
        v *= 4
    ai, ap, v = [ay.to(torch.bfloat16) for ay in (ai, ap, v)]
    if compress_ratio == 0:
        v = ap
        aw = 0
    elif compress_ratio in (1, 2, 4):
        aw = index_topk
    else:
        q = max(1, max(seq_lens_values) // 128)
        aw = max(128, 1 << (q - 1).bit_length())
    w = torch.full((r[-1], 128 + aw), -1, dtype=torch.int32)
    aa = torch.full((r[-1],), 128, dtype=torch.int32)
    for aj, (ac, u) in enumerate(zip(query_lens, seq_lens_values)):
        for x in range(ac):
            ak, av = (r[aj] + x, u - ac + x + 1)
            ax = [ar[aj][at // af] * af + at % af for at in range(max(0, av - 128), av)]
            w[ak, :len(ax)] = torch.tensor(ax, dtype=torch.int32)
            if compress_ratio:
                k = av // compress_ratio
                if compress_ratio == 128 or k <= index_topk:
                    am = torch.arange(k)
                elif index_pattern == 'score':
                    t = an[torch.tensor(o[aj], device=device)].reshape(-1, 8)
                    am = h(k, index_topk, index_pattern, t @ ai[ak, :, :8].float().mean(0))
                elif index_pattern == 'local':
                    am = torch.arange(k - index_topk, k)
                elif index_pattern == 'strided':
                    am = torch.linspace(0, k - 1, index_topk).long()
                elif index_pattern == 'shared_prefix':
                    am = torch.arange(index_topk)
                else:
                    am = torch.randperm(k)[:index_topk]
                m = [o[aj][at // p] * p + at % p for at in am.tolist()]
                w[ak, 128:128 + len(m)] = torch.tensor(m, dtype=torch.int32)
                aa[ak] += len(m)
    i(w[:, :128], af, name='swa')
    if aw:
        i(w[:, 128:], p, name='compressed')
    ao = torch.randn((num_heads,), dtype=torch.float32, device=device)
    if distribution == 'strong_sink':
        ao[::2] += 8
    ae = dict(query=ai, swa_kv_cache=ap, workspace_buffer=torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device), sparse_indices=w.to(device), compressed_kv_cache=v, sparse_topk_lens=aa.to(device), seq_lens=torch.tensor(seq_lens_values, dtype=torch.int32, device=device), out=torch.full_like(ai, float('nan')), bmm1_scale=s ** (-0.5), bmm2_scale=1.0, sinks=ao, kv_layout='HND', cum_seq_lens_q=torch.tensor(r, dtype=torch.int32, device=device), max_q_len=max(query_lens), enable_pdl=None, swa_topk_lens=None, extra_sparse_indices=None, extra_sparse_topk_lens=None)
    return {'query': ae['query'], 'swa_kv_cache': ae['swa_kv_cache'], 'workspace_buffer': ae['workspace_buffer'], 'sparse_indices': ae['sparse_indices'], 'compressed_kv_cache': ae.get('compressed_kv_cache', None), 'sparse_topk_lens': ae.get('sparse_topk_lens', None), 'seq_lens': ae.get('seq_lens', None), 'out': ae.get('out', None), 'bmm1_scale': ae.get('bmm1_scale', 1.0), 'bmm2_scale': ae.get('bmm2_scale', 1.0), 'sinks': ae.get('sinks', None), 'kv_layout': ae.get('kv_layout', 'HND'), 'cum_seq_lens_q': ae.get('cum_seq_lens_q', None), 'max_q_len': ae.get('max_q_len', None), 'enable_pdl': ae.get('enable_pdl', None), 'swa_topk_lens': ae.get('swa_topk_lens', None), 'extra_sparse_indices': ae.get('extra_sparse_indices', None), 'extra_sparse_topk_lens': ae.get('extra_sparse_topk_lens', None)}

def writable_views(x):
    k = {j: x[j] for j in ['out', 'workspace_buffer']}
    return k

def observable_inputs(x):
    k = writable_views(x)
    return {j: k[j] for j in ['out']}

def replay_tensors(x):
    return writable_views(x)

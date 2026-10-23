def a(values):
    b = []
    d = 0
    for j in values:
        d += j
        b.append(d)
    return b
import math
import torch

def e(nq, nk, heads, kv_heads, dim, value_dim, device, profile='normal', scale=None):
    assert heads % kv_heads == 0
    w = torch.arange(max(dim, value_dim), device=device).float()
    d = torch.sin((torch.arange(1, 9, device=device).float()[:, None] * w[None, :] + 0.3) / 17.0)
    d = d / d.square().mean(-1, keepdim=True).sqrt() / math.sqrt(8)
    u = torch.randn((kv_heads, 4, 8), device=device)
    j = u @ d[:, :dim]
    ag = u @ torch.cos((torch.arange(1, 9, device=device).float()[:, None] * w[None, :value_dim] + 0.7) / 19.0) / 2.0
    r = torch.empty((nk, kv_heads, dim), device=device)
    af = torch.empty((nk, kv_heads, value_dim), device=device)
    s = torch.linspace(0.7, 1.3, kv_heads, device=device)[None, :, None]
    k = torch.linspace(0.7, 1.3, dim, device=device)
    p = torch.arange(kv_heads, device=device)[None, :]
    for ad in range(0, nk, 1024):
        n = min(1024, nk - ad)
        ae = torch.randint(4, (n, kv_heads), device=device)
        r[ad:ad + n] = (0.8 * torch.randn((n, kv_heads, dim), device=device) + 0.6 * j[p, ae]) * s * k
        af[ad:ad + n] = (0.8 * torch.randn((n, kv_heads, value_dim), device=device) + 0.6 * ag[p, ae]) * s
    y = torch.empty((nq, heads, dim), device=device)
    z = j.repeat_interleave(heads // kv_heads, 0)
    ab = torch.arange(heads, device=device)[None, :]
    aa = torch.linspace(0.65, 1.35, heads, device=device)[None, :, None]
    for ad in range(0, nq, 1024):
        n = min(1024, nq - ad)
        ae = torch.randint(4, (n, heads), device=device)
        y[ad:ad + n] = (0.8 * torch.randn((n, heads, dim), device=device) + 0.6 * z[ab, ae]) * aa * k
    if dim == 576:
        y[..., 512:] *= 0.5
        r[..., 512:] *= 0.5
    ac = y[:min(nq, 8), ::max(1, heads // 4)].float()
    q = torch.arange(heads, device=device)[::max(1, heads // 4)] // (heads // kv_heads)
    t = r[:min(nk, 256), q].float()
    return (y, r, af)
'Construct the exact production call; all data preparation is outside forward.'
import torch
f = e

def g(shape, dtype, device):
    d = 1
    for b in shape:
        d *= b
    j = torch.full((d + 256,), -17, dtype=dtype, device=device)
    return j[128:-128].view(shape)

def h(query_lengths, kv_lengths, heads, kv_heads, head_dim, page_size, precision='bf16', causal=True, window_left=-1, sink=False, return_lse=True, distribution='normal', value_dim=None, device='cuda', q_scale_value=None, k_scale_value=None, v_scale_value=None):
    assert len(query_lengths) == len(kv_lengths) and heads % kv_heads == 0
    assert all((0 < y <= q for y, q in zip(query_lengths, kv_lengths)))
    p = value_dim or head_dim
    n = torch.float16 if precision == 'fp16' else torch.bfloat16
    z = torch.float8_e4m3fn if precision == 'fp8' else n
    r = torch.float8_e4m3fn if precision in ('fp8', 'mixed') else n
    aa, s, ah = (0.5 if precision == 'fp8' else 1.0, 0.25 if r == torch.float8_e4m3fn else 1.0, 0.75 if r == torch.float8_e4m3fn else 1.0)
    aa = aa if q_scale_value is None else q_scale_value
    s = s if k_scale_value is None else k_scale_value
    ah = ah if v_scale_value is None else v_scale_value
    j = [(u + page_size - 1) // page_size for u in kv_lengths]
    w = sum(j) + 1
    ac, ab, ad = f(sum(query_lengths), w * page_size, heads, kv_heads, head_dim, p, device, distribution, (192 if head_dim == 576 else head_dim) ** (-0.5))
    y = (ac / aa).to(z)
    q = (ab.reshape(w, page_size, kv_heads, head_dim).transpose(1, 2) / s).to(r)
    if head_dim == 576 and p == 512:
        ag = q[..., :512]
        ah = s
    else:
        ag = (ad.reshape(w, page_size, kv_heads, p).transpose(1, 2) / ah).to(r)
    del ac, ab, ad
    if distribution == 'tiny':
        y.copy_((y.float() * 0.03125).to(z))
    if distribution == 'peaked':
        y.copy_((y.float() * 3).clamp(-400, 400).to(z))
    if distribution == 'zero':
        ag.zero_()
    if distribution == 'balanced':
        y.zero_()
        ae = (torch.arange(page_size, device=device) % 2 * 2 - 1).float()
        ag.copy_((ae[None, None, :, None] / ah).expand_as(ag).to(r))
    v = torch.randperm(w - 1, device=device) + 1
    af = torch.zeros(len(j), max(j) + 1, dtype=torch.int32, device=device)
    k = 0
    for b, d in enumerate(j):
        af[b, :d] = v[k:k + d]
        k += d
    x = [0]
    for t in query_lengths:
        x.append(x[-1] + t)
    return dict(q=y, k=q, v=ag, out=g((y.shape[0], heads, p), n, device), cu_seqlens_q=torch.tensor(x, dtype=torch.int32, device=device), seq_lens=torch.tensor(kv_lengths, dtype=torch.int32, device=device), block_table=af, softmax_scale=(192 if head_dim == 576 else head_dim) ** (-0.5), q_scale=aa, k_scale=s, v_scale=ah, causal=causal, window_left=window_left, sinks=torch.linspace(-1, 4, heads, dtype=torch.float32, device=device) if sink else None, lse=g((y.shape[0], heads), torch.float32, device) if return_lse else None)

def i(cache, swizzle_v=False):
    z = cache.shape
    n = torch.tensor([448.0 / max(cache.float().abs().amax().item(), 1e-12)], dtype=torch.float32, device=cache.device)
    b = cache.float().reshape(*z[:-1], z[-1] // 16, 16)
    x = b.abs().amax(-1)
    y = (n * (x * (1.0 / 6.0))).to(torch.float8_e4m3fn)
    j = y.float() * n.reciprocal()
    k = torch.where(x != 0, j.reciprocal(), 0.0)
    ab = (b * k.unsqueeze(-1)).reshape(z)
    s = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    t = ab.abs()
    d = torch.zeros_like(ab, dtype=torch.uint8)
    for r in range(7):
        u = (s[r] + s[r + 1]) / 2
        d += (t > u if r % 2 == 0 else t >= u).to(torch.uint8)
    d |= torch.signbit(ab).to(torch.uint8) << 3
    v = (d[..., 0::2] | d[..., 1::2] << 4).contiguous()
    if swizzle_v:
        w, q, aa, p = y.shape
        y = y.view(torch.uint8).reshape(w, q, aa // 4, 4, 4, p // 4).permute(0, 1, 2, 4, 5, 3).reshape(w, q, aa, p).contiguous().view(torch.float8_e4m3fn)
    return (v, y, n.reciprocal().item())

def l(**kw):
    kw = dict(kw, precision='bf16')
    b = h(**kw)
    b['q_scale'] = 0.5
    b['q'] = (b['q'].float() / b['q_scale']).to(torch.float8_e4m3fn)
    b['k'], b['k_block_scales'], b['k_scale'] = i(b['k'])
    b['v'], b['v_block_scales'], b['v_scale'] = i(b['v'], swizzle_v=True)
    return b

def m(lengths, device):
    return torch.tensor([0] + a(lengths), dtype=torch.int32, device=device)

def o(**kw):
    t = l(**kw)
    q, d, r = (t['q'], t['k'], t['v'])
    p = t['cu_seqlens_q']
    j = t['seq_lens']
    b = q.device
    s = torch.zeros(128 * 1024 ** 2, dtype=torch.uint8, device=b)
    u = dict(query=q, workspace_buffer=s, seq_lens=j, bmm1_scale=t['softmax_scale'] * t['q_scale'] * t['k_scale'], bmm2_scale=t['v_scale'], out=t['out'], lse=t['lse'], return_lse=t['lse'] is not None, enable_pdl=False, backend='trtllm-gen')
    n = max(kw['query_lengths'])
    k = max(kw['kv_lengths'])
    u.update(kv_cache=d, block_tables=t['block_table'])
    u.update(kv_cache=(d, r), window_left=t['window_left'], sinks=t['sinks'], kv_layout='HND', out_dtype=t['out'].dtype)
    u.update(max_q_len=n, max_kv_len=k, batch_size=len(j), cum_seq_lens_q=p, cum_seq_lens_kv=m(kw['kv_lengths'], b), causal=t['causal'])
    if 'k_block_scales' in t:
        u['kv_cache_sf'] = (t['k_block_scales'], t['v_block_scales'])
    return u

def _make_inputs(head_dim, heads, kv_heads, kv_lengths, page_size, query_lengths, causal=None, distribution=None, precision=None, return_lse=None, sink=None, window_left=None, device='cuda'):
    j = {'device': device, **{b: k for b, k in {'causal': causal, 'distribution': distribution, 'head_dim': head_dim, 'heads': heads, 'kv_heads': kv_heads, 'kv_lengths': kv_lengths, 'page_size': page_size, 'precision': precision, 'query_lengths': query_lengths, 'return_lse': return_lse, 'sink': sink, 'window_left': window_left}.items() if k is not None}}
    d = o(**j)
    return {'query': d['query'], 'kv_cache': d['kv_cache'], 'workspace_buffer': d['workspace_buffer'], 'block_tables': d['block_tables'], 'seq_lens': d['seq_lens'], 'max_q_len': d['max_q_len'], 'max_kv_len': d['max_kv_len'], 'bmm1_scale': d['bmm1_scale'], 'bmm2_scale': d['bmm2_scale'], 'batch_size': d['batch_size'], 'cum_seq_lens_q': d['cum_seq_lens_q'], 'cum_seq_lens_kv': d['cum_seq_lens_kv'], 'window_left': d.get('window_left', -1), 'out': d.get('out', None), 'out_dtype': d.get('out_dtype', None), 'o_sf_scale': d.get('o_sf_scale', None), 'o_sf_vec_size': d.get('o_sf_vec_size', None), 'kv_layout': d.get('kv_layout', 'HND'), 'enable_pdl': d.get('enable_pdl', None), 'sinks': d.get('sinks', None), 'kv_cache_sf': d.get('kv_cache_sf', None), 'skip_softmax_threshold_scale_factor': d.get('skip_softmax_threshold_scale_factor', None), 'uses_shared_paged_kv_idx': d.get('uses_shared_paged_kv_idx', True), 'causal': d.get('causal', True), 'lse': d.get('lse', None), 'return_lse': d.get('return_lse', False), 'multi_ctas_kv_counter_buffer': d.get('multi_ctas_kv_counter_buffer', None), 'use_fp16_softmax': d.get('use_fp16_softmax', None), 'uses_spcompress': d.get('uses_spcompress', None), 'backend': d.get('backend', 'trtllm-gen')}

def writable_views(x):
    d = {b: x[b] for b in ['out', 'lse', 'workspace_buffer']}
    return d

def observable_inputs(x):
    d = writable_views(x)
    return {b: d[b] for b in ['out', 'lse']}

def replay_tensors(x):
    return writable_views(x)

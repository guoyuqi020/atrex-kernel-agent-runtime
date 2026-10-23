import math
import torch

def c(nq, nk, heads, kv_heads, dim, value_dim, device, profile='normal', scale=None):
    assert heads % kv_heads == 0
    s = torch.arange(max(dim, value_dim), device=device).float()
    d = torch.sin((torch.arange(1, 9, device=device).float()[:, None] * s[None, :] + 0.3) / 17.0)
    d = d / d.square().mean(-1, keepdim=True).sqrt() / math.sqrt(8)
    q = torch.randn((kv_heads, 4, 8), device=device)
    i = q @ d[:, :dim]
    ac = q @ torch.cos((torch.arange(1, 9, device=device).float()[:, None] * s[None, :value_dim] + 0.7) / 19.0) / 2.0
    n = torch.empty((nk, kv_heads, dim), device=device)
    ab = torch.empty((nk, kv_heads, value_dim), device=device)
    o = torch.linspace(0.7, 1.3, kv_heads, device=device)[None, :, None]
    j = torch.linspace(0.7, 1.3, dim, device=device)
    l = torch.arange(kv_heads, device=device)[None, :]
    for z in range(0, nk, 1024):
        k = min(1024, nk - z)
        aa = torch.randint(4, (k, kv_heads), device=device)
        n[z:z + k] = (0.8 * torch.randn((k, kv_heads, dim), device=device) + 0.6 * i[l, aa]) * o * j
        ab[z:z + k] = (0.8 * torch.randn((k, kv_heads, value_dim), device=device) + 0.6 * ac[l, aa]) * o
    u = torch.empty((nq, heads, dim), device=device)
    v = i.repeat_interleave(heads // kv_heads, 0)
    x = torch.arange(heads, device=device)[None, :]
    w = torch.linspace(0.65, 1.35, heads, device=device)[None, :, None]
    for z in range(0, nq, 1024):
        k = min(1024, nq - z)
        aa = torch.randint(4, (k, heads), device=device)
        u[z:z + k] = (0.8 * torch.randn((k, heads, dim), device=device) + 0.6 * v[x, aa]) * w * j
    if dim == 576:
        u[..., 512:] *= 0.5
        n[..., 512:] *= 0.5
    y = u[:min(nq, 8), ::max(1, heads // 4)].float()
    m = torch.arange(heads, device=device)[::max(1, heads // 4)] // (heads // kv_heads)
    p = n[:min(nk, 256), m].float()
    return (u, n, ab)
'Construct the exact production call; all data preparation is outside forward.'
import torch
e = c

def f(shape, dtype, device):
    d = 1
    for b in shape:
        d *= b
    i = torch.full((d + 256,), -17, dtype=dtype, device=device)
    return i[128:-128].view(shape)

def g(query_lengths, kv_lengths, heads, kv_heads, head_dim, page_size, precision='bf16', causal=True, window_left=-1, sink=False, return_lse=True, distribution='normal', value_dim=None, device='cuda', q_scale_value=None, k_scale_value=None, v_scale_value=None):
    assert len(query_lengths) == len(kv_lengths) and heads % kv_heads == 0
    assert all((0 < u <= m for u, m in zip(query_lengths, kv_lengths)))
    l = value_dim or head_dim
    k = torch.float16 if precision == 'fp16' else torch.bfloat16
    v = torch.float8_e4m3fn if precision == 'fp8' else k
    n = torch.float8_e4m3fn if precision in ('fp8', 'mixed') else k
    w, o, ad = (0.5 if precision == 'fp8' else 1.0, 0.25 if n == torch.float8_e4m3fn else 1.0, 0.75 if n == torch.float8_e4m3fn else 1.0)
    w = w if q_scale_value is None else q_scale_value
    o = o if k_scale_value is None else k_scale_value
    ad = ad if v_scale_value is None else v_scale_value
    i = [(q + page_size - 1) // page_size for q in kv_lengths]
    s = sum(i) + 1
    y, x, z = e(sum(query_lengths), s * page_size, heads, kv_heads, head_dim, l, device, distribution, (192 if head_dim == 576 else head_dim) ** (-0.5))
    u = (y / w).to(v)
    m = (x.reshape(s, page_size, kv_heads, head_dim).transpose(1, 2) / o).to(n)
    if head_dim == 576 and l == 512:
        ac = m[..., :512]
        ad = o
    else:
        ac = (z.reshape(s, page_size, kv_heads, l).transpose(1, 2) / ad).to(n)
    del y, x, z
    if distribution == 'tiny':
        u.copy_((u.float() * 0.03125).to(v))
    if distribution == 'peaked':
        u.copy_((u.float() * 3).clamp(-400, 400).to(v))
    if distribution == 'zero':
        ac.zero_()
    if distribution == 'balanced':
        u.zero_()
        aa = (torch.arange(page_size, device=device) % 2 * 2 - 1).float()
        ac.copy_((aa[None, None, :, None] / ad).expand_as(ac).to(n))
    r = torch.randperm(s - 1, device=device) + 1
    ab = torch.zeros(len(i), max(i) + 1, dtype=torch.int32, device=device)
    j = 0
    for b, d in enumerate(i):
        ab[b, :d] = r[j:j + d]
        j += d
    t = [0]
    for p in query_lengths:
        t.append(t[-1] + p)
    return dict(q=u, k=m, v=ac, out=f((u.shape[0], heads, l), k, device), cu_seqlens_q=torch.tensor(t, dtype=torch.int32, device=device), seq_lens=torch.tensor(kv_lengths, dtype=torch.int32, device=device), block_table=ab, softmax_scale=(192 if head_dim == 576 else head_dim) ** (-0.5), q_scale=w, k_scale=o, v_scale=ad, causal=causal, window_left=window_left, sinks=torch.linspace(-1, 4, heads, dtype=torch.float32, device=device) if sink else None, lse=f((u.shape[0], heads), torch.float32, device) if return_lse else None)

def h(**kw):
    o = g(**kw)
    m, d = (o['q'], o['k'])
    l = o['cu_seqlens_q']
    i = o['seq_lens']
    b = m.device
    n = torch.zeros(128 * 1024 ** 2, dtype=torch.uint8, device=b)
    p = dict(query=m, workspace_buffer=n, seq_lens=i, bmm1_scale=o['softmax_scale'] * o['q_scale'] * o['k_scale'], bmm2_scale=o['v_scale'], out=o['out'], lse=o['lse'], return_lse=o['lse'] is not None, enable_pdl=False, backend='trtllm-gen')
    k = max(kw['query_lengths'])
    j = max(kw['kv_lengths'])
    p.update(kv_cache=d, block_tables=o['block_table'])
    p.update(qk_nope_head_dim=128, kv_lora_rank=512, qk_rope_head_dim=64, max_seq_len=j, sinks=o['sinks'], return_lse_base='basee')
    if len(set(kw['query_lengths'])) == 1:
        p['query'] = m.view(len(i), k, *m.shape[1:])
        p['out'] = o['out'].view(len(i), k, *o['out'].shape[1:])
    else:
        p.update(cum_seq_lens_q=l, max_q_len=k)
    return p

def _make_inputs(head_dim, heads, kv_heads, kv_lengths, page_size, query_lengths, causal=None, precision=None, return_lse=None, sink=None, value_dim=None, window_left=None, device='cuda'):
    i = {'device': device, **{b: j for b, j in {'causal': causal, 'head_dim': head_dim, 'heads': heads, 'kv_heads': kv_heads, 'kv_lengths': kv_lengths, 'page_size': page_size, 'precision': precision, 'query_lengths': query_lengths, 'return_lse': return_lse, 'sink': sink, 'value_dim': value_dim, 'window_left': window_left}.items() if j is not None}}
    d = h(**i)
    return {'query': d['query'], 'kv_cache': d['kv_cache'], 'workspace_buffer': d['workspace_buffer'], 'qk_nope_head_dim': d['qk_nope_head_dim'], 'kv_lora_rank': d['kv_lora_rank'], 'qk_rope_head_dim': d['qk_rope_head_dim'], 'block_tables': d['block_tables'], 'seq_lens': d['seq_lens'], 'max_seq_len': d['max_seq_len'], 'sparse_mla_top_k': d.get('sparse_mla_top_k', 0), 'out': d.get('out', None), 'bmm1_scale': d.get('bmm1_scale', 1.0), 'bmm2_scale': d.get('bmm2_scale', 1.0), 'sinks': d.get('sinks', None), 'skip_softmax_threshold_scale_factor': d.get('skip_softmax_threshold_scale_factor', None), 'enable_pdl': d.get('enable_pdl', None), 'backend': d.get('backend', 'auto'), 'is_var_seq': d.get('is_var_seq', True), 'uses_shared_paged_kv_idx': d.get('uses_shared_paged_kv_idx', True), 'lse': d.get('lse', None), 'return_lse': d.get('return_lse', False), 'cute_dsl_impl': d.get('cute_dsl_impl', 'auto'), 'kv_scale_format': d.get('kv_scale_format', 'auto'), 'cum_seq_lens_q': d.get('cum_seq_lens_q', None), 'max_q_len': d.get('max_q_len', None), 'multi_ctas_kv_counter_buffer': d.get('multi_ctas_kv_counter_buffer', None), 'sparse_mla_top_k_lens': d.get('sparse_mla_top_k_lens', None), 'enable_dcp': d.get('enable_dcp', False), 'cp_world': d.get('cp_world', 1), 'cp_rank': d.get('cp_rank', 0), 'causal_seqlens_kv_global': d.get('causal_seqlens_kv_global', None), 'use_fp16_softmax': d.get('use_fp16_softmax', None), 'return_lse_base': d.get('return_lse_base', None)}

def writable_views(x):
    d = {b: x[b] for b in ['out', 'lse', 'workspace_buffer']}
    return d

def observable_inputs(x):
    d = writable_views(x)
    return {b: d[b] for b in ['out', 'lse']}

def replay_tensors(x):
    return writable_views(x)

import math
import torch

def f(nq, nk, heads, kv_heads, dim, value_dim, device, profile='normal', scale=None):
    assert heads % kv_heads == 0
    t = torch.arange(max(dim, value_dim), device=device).float()
    b = torch.sin((torch.arange(1, 9, device=device).float()[:, None] * t[None, :] + 0.3) / 17.0)
    b = b / b.square().mean(-1, keepdim=True).sqrt() / math.sqrt(8)
    q = torch.randn((kv_heads, 4, 8), device=device)
    d = q @ b[:, :dim]
    ao = q @ torch.cos((torch.arange(1, 9, device=device).float()[:, None] * t[None, :value_dim] + 0.7) / 19.0) / 2.0
    l = torch.empty((nk, kv_heads, dim), device=device)
    an = torch.empty((nk, kv_heads, value_dim), device=device)
    n = torch.linspace(0.7, 1.3, kv_heads, device=device)[None, :, None]
    e = torch.linspace(0.7, 1.3, dim, device=device)
    h = torch.arange(kv_heads, device=device)[None, :]
    for al in range(0, nk, 1024):
        g = min(1024, nk - al)
        am = torch.randint(4, (g, kv_heads), device=device)
        l[al:al + g] = (0.8 * torch.randn((g, kv_heads, dim), device=device) + 0.6 * d[h, am]) * n * e
        an[al:al + g] = (0.8 * torch.randn((g, kv_heads, value_dim), device=device) + 0.6 * ao[h, am]) * n
    x = torch.empty((nq, heads, dim), device=device)
    ah = d.repeat_interleave(heads // kv_heads, 0)
    aj = torch.arange(heads, device=device)[None, :]
    ai = torch.linspace(0.65, 1.35, heads, device=device)[None, :, None]
    for al in range(0, nq, 1024):
        g = min(1024, nq - al)
        am = torch.randint(4, (g, heads), device=device)
        x[al:al + g] = (0.8 * torch.randn((g, heads, dim), device=device) + 0.6 * ah[aj, am]) * ai * e
    if dim == 576:
        x[..., 512:] *= 0.5
        l[..., 512:] *= 0.5
    ak = x[:min(nq, 8), ::max(1, heads // 4)].float()
    k = torch.arange(heads, device=device)[::max(1, heads // 4)] // (heads // kv_heads)
    p = l[:min(nk, 256), k].float()
    return (x, l, an)
import torch
i = ['FP8_MAX_NORM', 'e8m0_ceil', 'e8m0_to_float', 'exp2_rcp', 'quantize_blocks', 'quantize_mxfp8_2d', 'swizzle_sf_rowwise', 'swizzle_sf_columnwise', 'quantize_to_mxfp8']
j = {torch.float8_e4m3fn: 448.0, torch.float8_e5m2: 57344.0}

def m(a, b):
    return (a + b - 1) // b

def o(v):
    v = v.float().contiguous()
    a = v.view(torch.int32)
    d = a >> 23 & 255
    e = a & 8388607
    g = (e > 0) & (d != 254) & ~((d == 0) & (e <= 4194304))
    b = d + g.to(torch.int32)
    b = torch.where(torch.isnan(v), torch.full_like(b, 255), b)
    b = torch.where(torch.isinf(v), torch.full_like(b, 254), b)
    return b.to(torch.uint8)

def r(e):
    a = e.to(torch.int32)
    b = torch.where(a == 254, torch.full_like(a, 4194304), 254 - a << 23)
    return b.view(torch.float32)

def u(e):
    return e.contiguous().view(torch.float8_e8m0fnu).float()

def w(x, fp8_dtype):
    g = j[fp8_dtype]
    x = x.float()
    a = x.abs().amax(dim=-1)
    e = torch.tensor(1.0 / g, dtype=torch.float32, device=x.device)
    d = o(a * e)
    h = r(d).unsqueeze(-1)
    b = (x * h).clamp_(-g, g).to(fp8_dtype)
    return (b, d)

def y(t2d, fp8_dtype):
    b, a = t2d.shape
    assert b % 128 == 0 and a % 128 == 0, f'pre-pad to 128 multiples, got {t2d.shape}'
    k = t2d.float()
    g, h = w(k.reshape(b, a // 32, 32), fp8_dtype)
    d, e = w(k.T.contiguous().reshape(a, b // 32, 32), fp8_dtype)
    return (g.reshape(b, a), h, d.reshape(a, b).T.contiguous(), e.transpose(0, 1).contiguous())

def z(sf):
    assert sf.dtype == torch.uint8
    *d, b, a = sf.shape
    assert b % 128 == 0 and a % 4 == 0, f'F8_128x4 needs R%128==0, C%4==0, got {sf.shape}'
    g = sf.reshape(*d, b // 128, 4, 32, a // 4, 4)
    e = len(d)
    g = g.permute(*range(e), e + 0, e + 3, e + 2, e + 1, e + 4)
    return g.contiguous().reshape(*d, b, a)

def aa(sf):
    return z(sf)

def ab(sf):
    a = z(sf.transpose(-1, -2).contiguous())
    return a.reshape(sf.shape)

def ac(tensor, b, h, s, d, block_size=32, fp8_dtype=torch.float8_e4m3fn, with_ref=True):
    assert block_size == 32, 'MXFP8 uses 32-element blocks'
    p = b * h
    k = m(m(d, block_size), 4) * 4
    g = k * block_size
    ai = m(m(s, block_size), 4) * 4
    ah = ai * block_size
    aq = tensor.float().reshape(p, s, d)
    q = g - d
    t = ah - s
    if t > 0 or q > 0:
        aq = torch.nn.functional.pad(aq, (0, q, 0, t))
    ap = aq.reshape(p * ah, g)
    v, x, a, e = y(ap, fp8_dtype)
    l = v.reshape(p, ah, g)[:, :s, :d].contiguous().reshape(b, h, s, d)
    ak = None
    if with_ref:
        aj = u(x)
        ak = torch.repeat_interleave(aj.reshape(p, ah, k), repeats=block_size, dim=2)[:, :s, :d].contiguous()
    al = aa(x)
    n = a.reshape(p, ah, g)[:, :s, :d].contiguous().reshape(b, h, s, d)
    an = None
    if with_ref:
        am = u(e)
        an = torch.repeat_interleave(am.reshape(p, ai, g), repeats=block_size, dim=1)[:, :s, :d].contiguous()
    ao = ab(e)
    return (l, ak, al, n, an, ao)
import torch
ad = f
ae = ac

def af(shape, dtype):
    b = 1
    for a in shape:
        b *= a
    return torch.full((b + 256,), -17, dtype=dtype, device='cuda')[128:-128].view(shape)

def ag(batch, heads, kv_heads, query_len, kv_len, head_dim, causal, window_left=None, distribution='normal', return_lse=True):
    n, k, p = ad(batch * query_len, batch * kv_len, heads, kv_heads, head_dim, head_dim, 'cuda', distribution)

    def operand(raw, h, s, columnwise=False):
        e = raw.reshape(batch, s, h, head_dim).transpose(1, 2).to(torch.bfloat16)
        if distribution == 'tiny':
            e *= 0.01
        if distribution == 'peaked':
            e *= 3
        if distribution == 'zero':
            e.zero_()
        b = ae(e, batch, h, s, head_dim, with_ref=False)
        a, d = (b[3], b[5]) if columnwise else (b[0], b[2])
        return (a.transpose(1, 2).contiguous().transpose(1, 2), d)
    h, x = operand(n, heads, query_len)
    d, v = operand(k, kv_heads, kv_len)
    aj, ah = operand(p, kv_heads, kv_len, True)
    g = af((batch, query_len, heads, head_dim), torch.bfloat16).transpose(1, 2)
    e = af((batch, heads, query_len), torch.float32) if return_lse else None
    q = head_dim ** (-0.5)
    s = dict(sample_q=h, sample_k=d, sample_v=aj, sample_o=g, sample_lse=e, is_causal=causal, causal_bottom_right=causal, window_size_left=window_left, scale_softmax=q, dtype_o=torch.bfloat16, pertensor_fp8=False, has_amax_o=False, seq_kv_lens_present=not causal)
    a = dict(q_tensor=h, k_tensor=d, v_tensor=aj, o_tensor=g, lse_tensor=e, sf_q=x, sf_k=v, sf_v=ah)
    if not causal:
        a['seq_kv_lens'] = torch.full((batch,), kv_len, device='cuda', dtype=torch.int32)
    return dict(setup=s, call=a, reference_options=dict(causal=causal, window_left=window_left, scale=q))

def _make_inputs(batch, head_dim, heads, kv_heads, kv_len, query_len, causal=None, distribution='normal', return_lse=True, window_left=None, device='cuda'):
    d = {'device': device, 'distribution': distribution, 'return_lse': return_lse, **{a: e for a, e in {'batch': batch, 'causal': causal, 'head_dim': head_dim, 'heads': heads, 'kv_heads': kv_heads, 'kv_len': kv_len, 'query_len': query_len, 'window_left': window_left}.items() if e is not None}}
    if device != 'cuda':
        raise ValueError('Native input generator requires CUDA')
    d.pop('device')
    b = ag(**d)['call']
    return {'q_tensor': b['q_tensor'], 'k_tensor': b['k_tensor'], 'v_tensor': b['v_tensor'], 'o_tensor': b['o_tensor'], 'lse_tensor': b.get('lse_tensor', None), 'sinks': b.get('sinks', None), 'seq_q_lens': b.get('seq_q_lens', None), 'seq_kv_lens': b.get('seq_kv_lens', None), 'scale_softmax': b.get('scale_softmax', None), 'current_stream': b.get('current_stream', None), 'sf_q': b.get('sf_q', None), 'sf_k': b.get('sf_k', None), 'sf_v': b.get('sf_v', None), 'amax_o': b.get('amax_o', None), 'descale_q': b.get('descale_q', None), 'descale_k': b.get('descale_k', None), 'descale_v': b.get('descale_v', None), 'scale_o': b.get('scale_o', None), 'workspace': b.get('workspace', None), 'block_table': b.get('block_table', None), 'block_table_v': b.get('block_table_v', None), 'gate': b.get('gate', None), 'sf_o': b.get('sf_o', None)}

def prepare_model(model_class, inputs, init_kwargs, input_kwargs):
    a = input_kwargs
    b = model_class(sample_q=inputs['q_tensor'], sample_k=inputs['k_tensor'], sample_v=inputs['v_tensor'], sample_o=inputs['o_tensor'], sample_lse=inputs['lse_tensor'], is_causal=a['causal'], causal_bottom_right=a['causal'], window_size_left=a.get('window_left'), scale_softmax=a['head_dim'] ** (-0.5), dtype_o=torch.bfloat16, pertensor_fp8=False, has_amax_o=False, seq_kv_lens_present=not a['causal'])
    assert b.check_support()
    b.compile()
    inputs['workspace'] = torch.empty(b.scratch_workspace_bytes(), dtype=torch.uint8, device=inputs['q_tensor'].device)
    return (b, inputs)

def writable_views(x):
    b = {a: x[a] for a in ['o_tensor', 'lse_tensor', 'workspace']}
    return b

def observable_inputs(x):
    b = writable_views(x)
    return {a: b[a] for a in ['o_tensor', 'lse_tensor']}

def replay_tensors(x):
    return writable_views(x)

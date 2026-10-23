import math
import torch
import torch.nn.functional as F

def e(raw, beta):
    a = torch.linspace(-0.4, 0.4, raw.shape[-1], device=raw.device)
    b = torch.randn((*raw.shape[:-1], 1), device=raw.device)
    raw.copy_(0.65 * b + math.sqrt(1 - 0.65 ** 2) * torch.randn_like(raw.float()) + a)
    beta.copy_(0.45 * b.squeeze(-1) + math.sqrt(1 - 0.45 ** 2) * torch.randn_like(beta.float()))

def f(p, shape, kind):
    bb = shape.get('domain', shape.get('distribution', 'normal'))
    if bb not in ('normal', 'core_buffer', 'output_only'):
        return p
    aq = kind.startswith('kda')
    ba = 'prefill' in kind
    bk = p.get('state', p.get('initial_state'))
    ad = bk.device
    an = bk.shape[1]
    x = bk.shape[-1]
    r = p.get('conv_state')
    ao = r is not None
    if 'metadata' in p:
        ax = shape['batch']
        j = list(range(ax + 1))
        bn = p['metadata']['non_spec_state_indices_tensor'].long()[:, None]
        b = [1] * ax
        bf = p['qkvz']
        ak = (bf.shape[-1] - 2 * an * x) // (2 * x)
    elif ba:
        at = shape['lengths']
        j = [0]
        for bh in at:
            j.append(j[-1] + bh)
        ax = len(at)
        bn = torch.arange(ax, device=ad)[:, None]
        b = [1] * ax
        ak = p['q'].shape[-2]
    else:
        j = p.get('cu_seqlens', p.get('query_start_loc')).tolist()
        ax = len(j) - 1
        bn = p.get('state_indices', p.get('ssm_state_indices')).long()
        if bn.ndim == 1:
            bn = bn[:, None]
        a = p.get('num_accepted_tokens')
        b = a.tolist() if a is not None else [1] * ax
        ak = an if aq else (p['mixed_qkv'].shape[1] - an * x) // (2 * x)
        bf = p.get('x', p.get('mixed_qkv'))
    bq = bn.shape[1]
    n = (2 * ak + an) * x
    am = 32
    if aq:
        d = p['A_log'].float().reshape(an)
        i = p.get('dt_bias', p.get('g_bias')).float().reshape(an, x)
        bo = p['weight'].float().permute(0, 2, 1).reshape(n, 4) if ao else None
        bg = p.get('raw_gate', p.get('raw_g', p.get('g')))
        be = p.get('raw_beta', p.get('beta'))
        if bg is not None and be is not None and shape.get('beta_is_logit', True):
            if 'g_bias' not in p or 'raw_beta' in p:
                e(bg, be)
    else:
        d = p['A_log'].float() if 'A_log' in p else torch.linspace(-3, 1, an, device=ad)
        i = p['dt_bias'].float() if 'dt_bias' in p else torch.linspace(-3, 3, an, device=ad).to(torch.bfloat16).float()
        bo = p['conv_weight'].float() if ao else None
    ai = torch.cat([torch.linspace(0.6, 1.4, al, device=ad)[:, None] * torch.linspace(0.7, 1.3, x, device=ad)[None, :] for al in (ak, ak, an)]).flatten()
    for ag in range(0, ax, 8):
        ay = list(range(ag, min(ag + 8, ax)))
        s = len(ay)
        t = max((j[bc + 1] - j[bc] for bc in ay))
        ah = min(t, 8) if ba else t
        bl = am + bq - 1
        bd = torch.randn((s, bl + ah + 3, n), device=ad) * 0.4 * ai
        for bm in range(1, bd.shape[1]):
            bd[:, bm] = 0.5 * bd[:, bm - 1] + math.sqrt(0.75) * bd[:, bm]
        bd = bd.to(torch.bfloat16).float()
        if ao:
            az = F.silu((bd.unfold(1, 4, 1) * bo[None, None, :, :]).sum(-1))
        else:
            az = F.silu(bd[:, 3:])
        k = torch.zeros((s, an, x, x), device=ad)
        for bm in range(bl):
            ar = az[:, bm, ak * x:2 * ak * x].reshape(s, ak, x)
            ar = ar * torch.rsqrt(ar.square().sum(-1, keepdim=True) + 1e-06)
            ar = ar.repeat_interleave(an // ak, 1)
            bp = az[:, bm, 2 * ak * x:].reshape(s, an, x)
            be = torch.randn((s, an), device=ad).to(torch.bfloat16).float()
            if aq:
                aj = (0.45 * be[:, :, None] + 0.8 * torch.randn((s, an, x), device=ad)).to(torch.bfloat16).float()
                br = d.exp()[None, :, None] * (aj + i)
                av = -d.exp()[None, :, None] * F.softplus(aj + i) if p.get('lower_bound', -5.0) is None else p.get('lower_bound', -5.0) * br.sigmoid()
                k *= av.exp()[:, :, None, :]
            else:
                aj = (0.45 * be + 0.8 * torch.randn_like(be)).to(torch.bfloat16).float()
                k *= torch.exp(-d.exp() * F.softplus(aj + i))[:, :, None, None]
            z = (bp - (k * ar[:, :, None, :]).sum(-1)) * be.sigmoid()[:, :, None]
            k += z[:, :, :, None] * ar[:, :, None, :]
            if bm >= am - 1:
                ap = bm - am + 1
                for au, bc in enumerate(ay):
                    ae = int(bn[bc, ap])
                    if ae > 0 or ba:
                        bk[ae].copy_(k[au].to(bk.dtype))
        if ao:
            for au, bc in enumerate(ay):
                bi = int(bn[bc, 0])
                bj = am + b[bc] - 1
                if bi <= 0:
                    continue
                r[bi, :, :bq + 2].copy_(bd[au, am:am + bq + 2].T.to(r.dtype))
                h, af = j[bc:bc + 2]
                bf[h:af, :n].copy_(bd[au, bj + 3:bj + 3 + af - h].to(bf.dtype))
    aw = shape.get('state_mode', shape.get('state_regime', 'continuation'))
    if aw == 'zero':
        bk.zero_()
    elif aw == 'mixed':
        for bc in range(0, ax, 4):
            bk[bn[bc]] = 0
            if ao and int(bn[bc, 0]) > 0:
                r[int(bn[bc, 0])].zero_()
    return p
import math
import torch
from torch.nn import functional as F

def g(heads, device):
    b = torch.linspace(0.6, 1.4, heads, device=device)
    a = torch.linspace(0.7, 1.3, 128, device=device)
    return b[:, None] * a[None, :]

def l(x, h, hv, postconv=False):
    d = 0
    for b in (h, h, hv, hv):
        a = d + b * 128
        if a > x.shape[-1]:
            break
        i = x[..., d:a].unflatten(-1, (b, 128))
        i.mul_(g(b, x.device))
        if postconv and d < (2 * h + hv) * 128:
            i.copy_(F.silu(i.float()))
        d = a

def m(a, b):
    d = torch.linspace(-0.65, 0.8, a.shape[-1], device=a.device)
    h = torch.randn(a.shape, device=a.device)
    b.copy_(d * h + (1 - d.square()).sqrt() * torch.randn_like(h))
    a.copy_(h)

def o(state, regime='continuation'):
    if regime not in ('continuation', 'mixed', 'zero'):
        raise ValueError('unknown state regime')
    if regime == 'zero':
        state.zero_()
        return
    i = state.shape[1]
    n = torch.exp(-torch.arange(16, device=state.device).float() / 4)
    n /= n.square().sum().sqrt()
    a = torch.linspace(0.1, 0.3, i, device=state.device)[None, :, None, None]
    h = torch.linspace(0.7, 1.3, 128, device=state.device)
    for b in range(0, state.shape[0], 8):
        d = state[b:b + 8]
        j = torch.randn((*d.shape[:2], 128, 16), device=state.device)
        k = torch.randn_like(j)
        p = j * n @ k.transpose(-1, -2)
        p *= h[None, None, :, None] * h[None, None, None, :]
        p *= a
        p += torch.randn_like(p) * 0.02
        d.copy_(p)
    if regime == 'mixed':
        state[::4].zero_()

def q(count, pool_size, pattern, device):
    if pool_size <= count:
        raise ValueError('pool must include distinct requested slots plus null slot 0')
    if pattern == 'scattered':
        return torch.randperm(pool_size - 1, device=device)[:count] + 1
    if pattern == 'high':
        return torch.arange(pool_size - count, pool_size, device=device)
    if pattern == 'contiguous':
        return torch.arange(1, count + 1, device=device)
    raise ValueError('unknown slot pattern')
'Qwen Flash post-conv MTP: packed QKVZ/BA views and paged state.'
import math
import torch
u = f
v = l
w = m
y = o
aa = q

def ab(shape, dtype, device, value=0):
    b = math.prod(shape)
    a = torch.full((b + 32,), value, dtype=dtype, device=device)
    return a[16:16 + b].view(shape)

def _make_inputs(tp, width, lengths, accepted_mode='cycle', domain='normal', state_dtype='bf16', device='cuda', head_ratio=3, state_regime='continuation', pool_size=2048, state_slot_padding=32, slot_pattern='scattered'):
    s, t = (16 // tp, 16 * head_ratio // tp)
    ad = len(lengths)
    an = sum(lengths)
    ai = (2 * s + t) * 128
    ag = ab((an, ai + t * 128), torch.bfloat16, device, 7)
    ag.normal_(0, 0.4)
    z = ag[:, :ai]
    r = ag[:, ai:].view(an, t, 128)
    v(ag, s, t, postconv=True)
    i = ab((an, 2 * t), torch.bfloat16, device, 9)
    i.normal_(0, 1)
    h, a = (i[:, :t], i[:, t:])
    w(a, h)
    d = ab((t,), torch.float32, device, 3)
    d.copy_(torch.linspace(-3, 1, t, device=device))
    n = ab((t,), torch.bfloat16, device, 4)
    n.copy_(torch.linspace(-3, 3, t, device=device))
    ao = ab((128,), torch.bfloat16, device, 5)
    ao.uniform_(0.7, 1.3)
    p = {'fp32': torch.float32, 'bf16': torch.bfloat16}[state_dtype]
    ah = pool_size
    ak = t * 128 * 128
    am = ak + state_slot_padding
    j = ab((ah * am,), p, device, 11)
    al = j.as_strided((ah, t, 128, 128), (am, 128 * 128, 128, 1))
    y(al, state_regime)
    x = ab((ad, width), torch.int32, device, 17)
    x.copy_(aa(ad * width, ah, slot_pattern, device).view(ad, width))
    b = ab((ad,), torch.int32, device, 19)
    b.copy_(torch.arange(ad, device=device) % width + 1 if accepted_mode == 'cycle' else torch.full((ad,), width, device=device))
    k = ab((ad + 1,), torch.int32, device, 23)
    k[0] = 0
    k[1:] = torch.tensor(lengths, device=device).cumsum(0)
    ae = ab((an, t, 128), torch.bfloat16, device, 13)
    ae.zero_()
    if domain == 'tiny':
        z[:, 2 * s * 128:] *= 1e-07
        al.mul_(1e-07)
    elif domain == 'zero':
        z.zero_()
        al.zero_()
    elif domain == 'qk_nearzero':
        z[:, :2 * s * 128] *= 1e-05
    elif domain == 'weak_decay':
        a.fill_(-20)
        h.fill_(-5)
    elif domain == 'strong_decay':
        a.fill_(20)
        h.fill_(8)
    elif domain == 'cancellation':
        z[:, :2 * s * 128:2] *= -8
        al[..., ::2] *= -8
    elif domain == 'gate_negative':
        r.fill_(-20)
    elif domain == 'beta_negative':
        h.fill_(-20)
    elif domain == 'large':
        z.mul_(20)
        al.mul_(20)
    elif domain == 'api_null':
        x[2].zero_()
        x[3, 0] = 0
    elif domain != 'normal':
        raise ValueError(domain)
    af = dict(mixed_qkv=z, a=a, b=h, A_log=d, dt_bias=n, state_indices=x, cu_seqlens=k, num_accepted_tokens=b, state=al, output_gate=r, norm_weight=ao, out=ae, scale=128 ** (-0.5), norm_eps=1e-06, output_gate_activation='sigmoid')
    aj = {'tp': tp, 'width': width, 'lengths': lengths, 'accepted_mode': accepted_mode, 'domain': domain, 'state_dtype': state_dtype, 'device': device, 'head_ratio': head_ratio, 'state_regime': state_regime, 'pool_size': pool_size, 'state_slot_padding': state_slot_padding, 'slot_pattern': slot_pattern}
    u(af, aj, 'gdn_postconv')
    return {'mixed_qkv': af['mixed_qkv'], 'a': af['a'], 'b': af['b'], 'A_log': af['A_log'], 'dt_bias': af['dt_bias'], 'state_indices': af['state_indices'], 'cu_seqlens': af['cu_seqlens'], 'num_accepted_tokens': af['num_accepted_tokens'], 'state': af['state'], 'output_gate': af['output_gate'], 'norm_weight': af['norm_weight'], 'out': af.get('out', None), 'scale': af.get('scale', 128 ** (-0.5)), 'norm_eps': af.get('norm_eps', 1e-05), 'output_gate_activation': af.get('output_gate_activation', 'silu')}

def ac(x):
    h = x['cu_seqlens'].tolist()
    j = x['state_indices'].tolist()
    b = x['num_accepted_tokens'].tolist()
    p = []
    k = []
    for i, (r, a) in enumerate(zip(j, b)):
        d = h[i + 1] - h[i]
        if d <= 0:
            continue
        if d > 8 or not 0 < a <= len(r) or r[a - 1] <= 0:
            k.extend(range(h[i], h[i + 1]))
            continue
        p.extend((n for n in r[:d] if n > 0))
    return (sorted(set(p)), k)

def writable_views(x):
    i = {b: x[b] for b in ['state', 'out']}
    h, a = ac(x)
    i['state'] = [x['state'][d] for d in h]
    return i

def observable_inputs(x):
    b = writable_views(x)
    return {a: b[a] for a in ['state', 'out']}

def replay_tensors(x):
    return writable_views(x)

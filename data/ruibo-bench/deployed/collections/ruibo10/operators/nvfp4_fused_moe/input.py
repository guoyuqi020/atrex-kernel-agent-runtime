import math
import torch

def d(shape, device, std=1.0, role='projection'):
    i = torch.randn(shape, device=device)
    a = torch.linspace(0.65, 1.35, shape[-1], device=device)
    i.mul_(a)
    if role == 'fc2':
        e = torch.randn_like(i)
        h = 0.25 * e + math.sqrt(1 - 0.25 ** 2) * i
        i = torch.nn.functional.silu(e) * h
    elif role == 'weight':
        i.mul_(torch.linspace(0.7, 1.3, shape[-2], device=device)[:, None])
    elif role != 'projection':
        raise ValueError(role)
    i.mul_(std)
    return i

def f(shape, tokens, experts, topk, device, rank=0, world=1):
    assert 0 < topk <= experts and experts % world == 0
    p = shape.get('routing', 'balanced')
    i = torch.randn(((tokens + 7) // 8, 8), device=device)
    i = i.repeat_interleave(8, 0)[:tokens]
    i = 0.6 * i + 0.8 * torch.randn((tokens, 8), device=device)
    q = torch.randn((8, experts), device=device) / math.sqrt(8)
    n = i @ q + 0.35 * torch.randn((tokens, experts), device=device)
    h = torch.arange(experts, device=device)
    if p == 'skewed':
        n += 1.5 * torch.cos(h.float() * 2 * math.pi / experts)
    elif p == 'hotspot':
        n[:, :topk] += 32.0
    elif p in ('remote', 'local'):
        assert world > 1
        a = (rank + 1) % world if p == 'remote' else rank
        n += (h // (experts // world) == a).float() * 4.0
    elif p not in ('balanced', 'uniform', 'empty'):
        raise ValueError('unknown routing profile: ' + p)
    s, j = n.topk(topk, dim=-1)
    x = s.softmax(-1)
    if p == 'empty':
        j.fill_(-1)
    if shape.get('masked_routes'):
        j[::3, -1] = -1
    v = shape.get('rank_lengths', [tokens] * world)[rank]
    j[v:] = -1
    x.masked_fill_(j < 0, 0.0)
    t = j[j >= 0]
    k = torch.bincount(t, minlength=experts)
    e = torch.bincount(t // (experts // world), minlength=world)
    return (j, x)
'Canonical stored-precision inputs; no dependency on a serving framework.'
import math
import torch
g = d
l = f
m = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

def o(shape, dtype, device):
    e = math.prod(shape)
    a = torch.full((e + 128,), -17, dtype=dtype, device=device)
    return a[64:-64].view(shape)

def r(value, block, global_scale=1.0, scale_rows=1):
    n = value.shape[-1]
    assert n % block == 0
    j = value.float().reshape(*value.shape[:-1], n // block, block)
    v = j.abs().amax(-1).clamp_min(2.0 ** (-20))
    if scale_rows != 1:
        assert value.shape[-2] % scale_rows == 0
        v = v.reshape(*v.shape[:-2], value.shape[-2] // scale_rows, scale_rows, v.shape[-1]).amax(-2)
    a = 6.0
    x = v / a / global_scale
    aa = x.clamp(2.0 ** (-9), 448).to(torch.float8_e4m3fn)
    x = aa.float()
    i = x.repeat_interleave(scale_rows, -2)
    s = (j / (i.unsqueeze(-1) * global_scale)).reshape_as(value)
    p = s.abs()
    e = torch.zeros_like(p, dtype=torch.uint8)
    for k in range(7):
        q = (m[k] + m[k + 1]) / 2
        h = p > q if k % 2 == 0 else p >= q
        e += h.to(torch.uint8)
    e |= torch.signbit(s).to(torch.uint8) << 3
    t = e[..., 0::2] | e[..., 1::2] << 4
    return (t, aa)

def u(shape, block, device, std=1.0, zero=False, global_scale=1.0, scale_rows=1):
    if len(shape) == 3:
        e = [u(shape[1:], block, device, std, zero, global_scale, scale_rows) for a in range(shape[0])]
        return (torch.stack([i[0] for i in e]), None if e[0][1] is None else torch.stack([i[1] for i in e]))
    h = g(shape, device, std, role='weight' if std != 1.0 else 'projection')
    if zero:
        h.zero_()
    return r(h, block, global_scale, scale_rows)

def w(scale):
    a, h, e = scale.shape
    j, i = ((h + 127) // 128 * 128, (e + 3) // 4 * 4)
    k = torch.zeros((a, j, i), dtype=torch.uint8, device=scale.device)
    k[:, :h, :e] = scale.view(torch.uint8)
    return k.reshape(a, j // 128, 4, 32, i // 4, 4).permute(0, 1, 4, 3, 2, 5).contiguous().reshape(a, j // 128, i // 4, 32, 16).view(scale.dtype)

def y(q, sf, gated):
    a = q.shape[1]
    h = torch.arange(a, device=q.device)
    h = h // 32 * 32 + h % 8 * 4 + h % 32 // 8
    if gated:
        h = h // 2 + h % 2 * (a // 2)
    e = q[:, h].contiguous()
    i = w(sf.view(torch.uint8)[:, h]).reshape_as(sf).view(sf.dtype)
    return (e, i)

def z(s, seed=0, device='cuda'):
    torch.manual_seed(seed)
    aa, i, j, h, n = map(s.__getitem__, ('tokens', 'hidden', 'intermediate', 'local_experts', 'topk'))
    e = 16
    a, t = u((aa, i), 16, device, zero=s.get('distribution') == 'zero')
    ab, p = u((h, 2 * j, i), e, device, std=i ** (-0.5))
    ac, q = u((h, i, j), e, device, std=(j * s.get('expert_tp', 1)) ** (-0.5))
    ab, p = y(ab, p, True)
    ac, q = y(ac, q, False)
    k, v = l(s, aa, h, n, device)
    k = k.to(torch.int32)
    x = s.get('activation') == 'situ'
    return dict(topk_ids=(k, v), routing_bias=None, hidden_states=a, hidden_states_scale=t, gemm1_weights=ab, gemm1_weights_scale=p, gemm1_bias=None, gemm1_alpha=torch.full((h,), 4.0 if x else s.get('activation_alpha', 1.0), dtype=torch.float32, device=device), gemm1_beta=torch.full((h,), 25.0 if x else s.get('activation_beta', 0.0), dtype=torch.float32, device=device), gemm1_clamp_limit=None if s.get('activation_limit') is None else torch.full((h,), s['activation_limit'], dtype=torch.float32, device=device), gemm2_weights=ac, gemm2_weights_scale=q, gemm2_bias=None, output1_scale_scalar=torch.ones(h, device=device), output1_scale_gate_scalar=torch.ones(h, device=device), output2_scale_scalar=torch.ones(h, device=device), num_experts=h, top_k=n, n_group=None, topk_group=None, intermediate_size=j, local_expert_offset=0, local_num_experts=h, routed_scaling_factor=1.0, routing_method_type=0, do_finalize=True, enable_pdl=False, activation_type=10 if x else 3, per_token_scale=None, output=o((aa, i), torch.bfloat16, device), tune_max_num_tokens=8192, gemm1_lora_delta=None, valid_hidden_size=None, valid_intermediate_size=None, num_fused_shared_experts=None, hidden_states_scale_layout=2)

def _make_inputs(hidden, intermediate, local_experts, tokens, topk, activation=None, activation_alpha=None, activation_beta=None, activation_limit=None, distribution=None, masked_routes=None, routing=None, model_geometry=None, expert_tp=None, device='cuda'):
    h = {'device': device, **{a: i for a, i in {'activation': activation, 'activation_alpha': activation_alpha, 'activation_beta': activation_beta, 'activation_limit': activation_limit, 'distribution': distribution, 'hidden': hidden, 'intermediate': intermediate, 'local_experts': local_experts, 'masked_routes': masked_routes, 'routing': routing, 'tokens': tokens, 'topk': topk, 'model_geometry': model_geometry, 'expert_tp': expert_tp}.items() if i is not None}}
    e = z({a: i for a, i in h.items() if a != 'device'}, seed=torch.initial_seed(), device=device)
    return {'topk_ids': e['topk_ids'], 'routing_bias': e['routing_bias'], 'hidden_states': e['hidden_states'], 'hidden_states_scale': e['hidden_states_scale'], 'gemm1_weights': e['gemm1_weights'], 'gemm1_weights_scale': e['gemm1_weights_scale'], 'gemm1_bias': e['gemm1_bias'], 'gemm1_alpha': e['gemm1_alpha'], 'gemm1_beta': e['gemm1_beta'], 'gemm1_clamp_limit': e['gemm1_clamp_limit'], 'gemm2_weights': e['gemm2_weights'], 'gemm2_weights_scale': e['gemm2_weights_scale'], 'gemm2_bias': e['gemm2_bias'], 'output1_scale_scalar': e['output1_scale_scalar'], 'output1_scale_gate_scalar': e['output1_scale_gate_scalar'], 'output2_scale_scalar': e['output2_scale_scalar'], 'num_experts': e['num_experts'], 'top_k': e['top_k'], 'n_group': e['n_group'], 'topk_group': e['topk_group'], 'intermediate_size': e['intermediate_size'], 'local_expert_offset': e['local_expert_offset'], 'local_num_experts': e['local_num_experts'], 'routed_scaling_factor': e['routed_scaling_factor'], 'routing_method_type': e.get('routing_method_type', 0), 'do_finalize': e.get('do_finalize', True), 'enable_pdl': e.get('enable_pdl', None), 'activation_type': e.get('activation_type', 3), 'per_token_scale': e.get('per_token_scale', None), 'output': e.get('output', None), 'tune_max_num_tokens': e.get('tune_max_num_tokens', 8192), 'gemm1_lora_delta': e.get('gemm1_lora_delta', None), 'valid_hidden_size': e.get('valid_hidden_size', None), 'valid_intermediate_size': e.get('valid_intermediate_size', None), 'num_fused_shared_experts': e.get('num_fused_shared_experts', None), 'hidden_states_scale_layout': e.get('hidden_states_scale_layout', None)}

def writable_views(x):
    e = {a: x[a] for a in ['output']}
    return e

def observable_inputs(x):
    e = writable_views(x)
    return {a: e[a] for a in ['output']}

def replay_tensors(x):
    return writable_views(x)

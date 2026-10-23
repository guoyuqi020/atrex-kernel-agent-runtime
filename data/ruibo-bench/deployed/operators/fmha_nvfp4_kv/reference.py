def _prefix_sums(values):
    result = []
    total = 0
    for value in values:
        total += value
        result.append(total)
    return result
'Independent FP32 reference over the actual stored production inputs.\n\n**version**: nvfp4-kv-to-e4m3-v1\n\n**payload**: Decode actual packed E2M1 nibbles and E4M3 group16 scales; V uses documented 4-token interleave inverse.\n\n**kv_transform**: RNE E4M3FN finite saturation at448 after payload*block_scale, before global descales. Global Q/K factors and softmax scale remain bmm1_scale; V global descale remains bmm2_scale.\n\n**arithmetic**: Independent FP32 QK, softmax and PV with TF32 disabled; preserve existing FP32 P scoring policy, not a claim of reproducing every internal native P approximation.\n\n**outputs**: BF16 output buffer and optional log2 LSE, existing full output/return/alias/nonwrite storage checks.\n\n**source**: Pinned d849eca46875: trtllm/fmha/kernelParams.h transformsKv converts KV to Q dtype E4M3 before MMA.\n\n**native_defect**: HD256 V scale address bug stays a native failure; do not model observed wrong addresses in scoring.\n\nNumerical comparison uses the supplied standard runner config. No native\nkernel or hidden prequantization tensor supplies the scoring truth.\n'
import math
import torch

def _attention(q, k, v, out, cu_seqlens_q, seq_lens, block_table, softmax_scale, q_scale, k_scale, v_scale, causal, window_left, sinks, lse):
    ptr = cu_seqlens_q.tolist()
    page = k.shape[2]
    mapping = torch.arange(q.shape[1], device=q.device) // (q.shape[1] // k.shape[1])
    for b, length in enumerate(seq_lens.tolist()):
        first, last = ptr[b:b + 2]
        qlen = last - first
        for start in range(0, qlen, 32):
            end = min(start + 32, qlen)
            query = q[first + start:first + end].float() * q_scale
            visible = length - qlen + torch.arange(start, end, device=q.device)
            maximum = torch.full(query.shape[:2], -torch.inf, device=q.device)
            denom = torch.zeros_like(maximum)
            accum = torch.zeros(*query.shape[:2], v.shape[-1], device=q.device)
            for left in range(0, length, 2048):
                positions = torch.arange(left, min(length, left + 2048), device=q.device)
                pages = block_table[b, positions // page].long()
                key = k[pages, :, positions % page][:, mapping].float() * k_scale
                value = v[pages, :, positions % page][:, mapping].float() * v_scale
                score = torch.einsum('qhd,khd->qhk', query, key) * softmax_scale
                allowed = torch.ones((end - start, len(positions)), dtype=torch.bool, device=q.device)
                if causal:
                    allowed &= positions[None] <= visible[:, None]
                if window_left >= 0:
                    allowed &= positions[None] >= visible[:, None] - window_left
                score.masked_fill_(~allowed[:, None], -torch.inf)
                next_max = torch.maximum(maximum, score.amax(-1))
                safe = torch.where(torch.isfinite(next_max), next_max, 0.0)
                rescale = (maximum - safe).exp()
                weights = (score - safe[..., None]).exp()
                accum = accum * rescale[..., None] + torch.einsum('qhk,khd->qhd', weights, value)
                denom = denom * rescale + weights.sum(-1)
                maximum = next_max
            base_lse = maximum + denom.log()
            output = accum / denom[..., None]
            total_lse = base_lse
            if sinks is not None:
                total_lse = torch.logaddexp(base_lse, sinks[None])
                output *= (base_lse - total_lse).exp()[..., None]
            out[first + start:first + end].copy_(output)
            if lse is not None:
                lse[first + start:first + end].copy_(total_lse)
    return (out, lse)

def decode(payload, scales, is_v):
    p, h, t, packed = payload.shape
    groups = packed // 8
    sf = scales.float()
    if is_v:
        sf = sf.reshape(p, h, t // 4, groups, 4).permute(0, 1, 2, 4, 3).reshape(p, h, t, groups)
    code = torch.stack((payload & 15, payload >> 4), -1).flatten(-2).long()
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device=payload.device)
    return lut[code] * sf.repeat_interleave(16, -1)

def pointer(lengths, device):
    return torch.tensor([0] + _prefix_sums(lengths), dtype=torch.int32, device=device)

def normalized(x):
    kind = 'context_nvfp4'
    q = x['query']
    lens = x['seq_lens']
    out = x['out']
    k, v = x['kv_cache']
    table = x['block_tables']
    ptr = x.get('cum_seq_lens_q')
    if q.ndim == 4:
        ptr = pointer([q.shape[1]] * q.shape[0], q.device)
        q = q.flatten(0, 1)
        out = out.flatten(0, 1)
    ks, vs = x['kv_cache_sf']
    k = decode(k, ks, False)
    v = decode(v, vs, True)
    return dict(q=q, k=k, v=v, out=out, cu_seqlens_q=ptr, seq_lens=lens, block_table=table, softmax_scale=x['bmm1_scale'], q_scale=1.0, k_scale=1.0, v_scale=x['bmm2_scale'], causal=x.get('causal', x.get('is_causal', True)), window_left=x.get('window_left', -1), sinks=x.get('sinks', x.get('attention_sinks')), lse=x['lse'])

def returned(x):
    return (x['out'], x['lse']) if x.get('return_lse') else x['out']

class Model(torch.nn.Module):

    def __init__(self, **prepared):
        super().__init__()
        self.prepared = prepared

    def forward(self, query, kv_cache, workspace_buffer, block_tables, seq_lens, max_q_len, max_kv_len, bmm1_scale, bmm2_scale, batch_size, cum_seq_lens_q, cum_seq_lens_kv, window_left=-1, out=None, out_dtype=None, o_sf_scale=None, o_sf_vec_size=None, kv_layout='HND', enable_pdl=None, sinks=None, kv_cache_sf=None, skip_softmax_threshold_scale_factor=None, uses_shared_paged_kv_idx=True, causal=True, lse=None, return_lse=False, multi_ctas_kv_counter_buffer=None, use_fp16_softmax=None, uses_spcompress=None, backend='trtllm-gen'):
        x = {'query': query, 'kv_cache': kv_cache, 'workspace_buffer': workspace_buffer, 'block_tables': block_tables, 'seq_lens': seq_lens, 'max_q_len': max_q_len, 'max_kv_len': max_kv_len, 'bmm1_scale': bmm1_scale, 'bmm2_scale': bmm2_scale, 'batch_size': batch_size, 'cum_seq_lens_q': cum_seq_lens_q, 'cum_seq_lens_kv': cum_seq_lens_kv, 'window_left': window_left, 'out': out, 'out_dtype': out_dtype, 'o_sf_scale': o_sf_scale, 'o_sf_vec_size': o_sf_vec_size, 'kv_layout': kv_layout, 'enable_pdl': enable_pdl, 'sinks': sinks, 'kv_cache_sf': kv_cache_sf, 'skip_softmax_threshold_scale_factor': skip_softmax_threshold_scale_factor, 'uses_shared_paged_kv_idx': uses_shared_paged_kv_idx, 'causal': causal, 'lse': lse, 'return_lse': return_lse, 'multi_ctas_kv_counter_buffer': multi_ctas_kv_counter_buffer, 'use_fp16_softmax': use_fp16_softmax, 'uses_spcompress': uses_spcompress, 'backend': backend}
        torch.backends.cuda.matmul.allow_tf32 = False
        z = normalized(x)
        for key in ('k', 'v'):
            z[key] = z[key].clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
        target = z['out']
        target_lse = z['lse']
        z['out'] = torch.empty_like(target, dtype=torch.float32)
        if target_lse is not None:
            z['lse'] = torch.empty_like(target_lse)
        _attention(**z)
        out.copy_(z['out'].reshape_as(out))
        if target_lse is not None:
            lse.copy_((z['lse'] / math.log(2)).reshape_as(lse))
        return returned(x)

"""Independent FP32 reference over the actual stored production inputs.

**version**: explicit-mxfp8-probability-v1

**graph**: Decode actual E4M3 Q/K/V and E8M0 block32 scales → FP32 QK and mask → FP8 probability QDQ → FP32 PV accumulation and unquantized softmax denominator → BF16 output; FP32 LSE uses unquantized scores.

**quantization**: Probability format E4M3FN, nearest ties-to-even, saturate448. P MMA scale is unit E8M0. D128 casts16*exp(score-normalizer) and accumulates denominator in the same16x units; D256 casts exp(score-normalizer).

**grouping**: Each row visits128-key groups in increasing key order. Initial normalizer=-inf; update to current group maximum only when it exceeds old normalizer by more than4*ln(2). On update, rescale accumulated numerator and denominator by exp(old-new). Quantized P contributes to numerator; FP32 unrounded exp contributes to denominator. Fully masked groups contribute zero; all-zero denominator gives zero output.

**layout**: Q/K row block32, V sequence-axis block32 using the corrected dense F8_128x4 E8M0 layout. Native splits/scheduling may differ but must satisfy the same output tolerance; this numerical grouping does not require a particular CTA or launch structure.

**source**: nvidia-cudnn-frontend1.30.0: sm100/prefill_d128_mxfp8.py P_CAST_LOG2_SCALE=4; prefill_d256_mxfp8.py unshifted P E4M3 cast; config_sm100.py rescale_threshold(FP8)=4. Existing input/model/masks unchanged.

**arithmetic**: Independent FP32 between QDQ; TF32 disabled. No native quantizer, native internal tensor, pre-quantized hidden input or FP64 scoring truth.

Numerical comparison uses the supplied standard runner config. No native
kernel or hidden prequantization tensor supplies the scoring truth.
"""
import math
'Independent FP32 oracle from E4M3 payload and actual F8_128x4 E8M0 bytes.'
import torch

def attention_p(logits, values, head_dim):
    """MXFP8 P QDQ:128-key groups, delayed rescale4 log2 units.

    D128 carries16*exp in both numerator and unrounded denominator;
    D256 carries exp. Constant unit E8M0 P MMA scales, E4M3 casts.
    Empty masked groups contribute zero; no candidate schedule is imposed.
    """
    bias = 16.0 if head_dim == 128 else 1.0
    maximum = torch.full_like(logits[:, :1], -torch.inf)
    denom = torch.zeros_like(maximum)
    numerator = torch.zeros((logits.shape[0], values.shape[-1]), device=logits.device)
    for left in range(0, logits.shape[-1], 128):
        score = logits[:, left:left + 128]
        block_max = score.amax(-1, keepdim=True)
        update = block_max - maximum > 4.0 * math.log(2.0)
        new_max = torch.where(update, block_max, maximum)
        live = torch.isfinite(new_max)
        safe = torch.where(live, new_max, 0.0)
        alpha = torch.where(torch.isfinite(maximum), (maximum - safe).exp(), 0.0)
        probability = (score - safe).exp() * bias
        rounded = probability.clamp(0.0, 448.0).to(torch.float8_e4m3fn).float()
        numerator = numerator * alpha + rounded @ values[left:left + 128]
        denom = denom * alpha + probability.sum(-1, keepdim=True)
        maximum = new_max
    return torch.where(denom > 0, numerator / denom, 0.0)

def decode(data, sf, columnwise=False):
    b, h, s, d = data.shape
    rows, cols = (d, (s + 31) // 32) if columnwise else (s, (d + 31) // 32)
    rp = (rows + 127) // 128 * 128
    cp = (cols + 3) // 4 * 4
    r = torch.arange(rows, device=data.device)[:, None]
    c = torch.arange(cols, device=data.device)[None, :]
    offset = r // 128 * (cp // 4) * 512 + c // 4 * 512 + r % 32 * 16 + r % 128 // 32 * 4 + c % 4
    if columnwise:
        group = torch.arange(b * h, device=data.device).reshape(b, h, 1, 1)
        offset = r // 128 * (b * h * cp // 4) * 512 + (group * (cp // 4) + c // 4) * 512 + r % 32 * 16 + r % 128 // 32 * 4 + c % 4
        exponent = sf.reshape(-1)[offset].int() - 127
    else:
        exponent = sf.reshape(b, h, rp * cp)[:, :, offset].int() - 127
    scales = torch.exp2(exponent.float())
    scales = scales.repeat_interleave(32, -1)
    if columnwise:
        scales = scales[..., :s].transpose(-1, -2)
    else:
        scales = scales[..., :d]
    return data.float() * scales

def compute(payload):
    x = payload['call']
    opts = payload['reference_options']
    q = decode(x['q_tensor'], x['sf_q'])
    k = decode(x['k_tensor'], x['sf_k'])
    v = decode(x['v_tensor'], x['sf_v'], True)
    b, h, s, d = q.shape
    sk = k.shape[2]
    group = h // k.shape[1]
    out = torch.empty_like(x['o_tensor'])
    lse = torch.empty((b, h, s), dtype=torch.float32, device=q.device)
    pos = torch.arange(sk, device=q.device)[None, :]
    for bi in range(b):
        for hi in range(h):
            for start in range(0, s, 128):
                end = min(start + 128, s)
                logits = q[bi, hi, start:end] @ k[bi, hi // group].T * opts['scale']
                center = torch.arange(start, end, device=q.device)[:, None] + sk - s
                if opts['causal']:
                    logits.masked_fill_(pos > center, -torch.inf)
                if opts['window_left'] is not None:
                    logits.masked_fill_(pos < center - opts['window_left'], -torch.inf)
                lse[bi, hi, start:end] = logits.logsumexp(-1)
                out[bi, hi, start:end] = attention_p(logits, v[bi, hi // group], d).to(out.dtype)
    result = {'o_tensor': out}
    if x['lse_tensor'] is not None:
        result['lse_tensor'] = lse
    return result

class Model(torch.nn.Module):

    def __init__(self, **setup):
        super().__init__()
        self.options = dict(causal=setup.get('is_causal', False), window_left=setup.get('window_size_left'), scale=setup.get('scale_softmax'))

    def check_support(self):
        return True

    def compile(self):
        return None

    def scratch_workspace_bytes(self):
        return 0

    def forward(self, q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor=None, sinks=None, seq_q_lens=None, seq_kv_lens=None, scale_softmax=None, current_stream=None, sf_q=None, sf_k=None, sf_v=None, amax_o=None, descale_q=None, descale_k=None, descale_v=None, scale_o=None, workspace=None, block_table=None, block_table_v=None, gate=None, sf_o=None):
        call = {'q_tensor': q_tensor, 'k_tensor': k_tensor, 'v_tensor': v_tensor, 'o_tensor': o_tensor, 'lse_tensor': lse_tensor, 'sinks': sinks, 'seq_q_lens': seq_q_lens, 'seq_kv_lens': seq_kv_lens, 'scale_softmax': scale_softmax, 'current_stream': current_stream, 'sf_q': sf_q, 'sf_k': sf_k, 'sf_v': sf_v, 'amax_o': amax_o, 'descale_q': descale_q, 'descale_k': descale_k, 'descale_v': descale_v, 'scale_o': scale_o, 'workspace': workspace, 'block_table': block_table, 'block_table_v': block_table_v, 'gate': gate, 'sf_o': sf_o}
        torch.backends.cuda.matmul.allow_tf32 = False
        options = dict(self.options)
        if options['scale'] is None:
            options['scale'] = float(scale_softmax) if scale_softmax is not None else q_tensor.shape[-1] ** (-0.5)
        values = compute(dict(call=call, reference_options=options))
        for key, value in values.items():
            call[key].copy_(value)
        return None

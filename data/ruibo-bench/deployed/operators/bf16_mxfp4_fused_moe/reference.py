"""FP32 oracle from production packed weights; permutation helpers decode layout only."""
import torch

def decode(q, sf, fmt):
    block = 32
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device=q.device)
    value = lut[torch.stack((q & 15, q >> 4), -1).long()].flatten(-2)
    scale = torch.exp2(sf.view(torch.uint8).float() - 127)
    return value * scale.repeat_interleave(block, -1)

def weight(q, sf, gated):
    """Undo published gate/up row interleave, row shuffle and scale swizzle.

    With epilogue_tile_m=128, each32-row block sends row j to
    8*(j%4)+j//4. Scale rows use the same map as payload rows.
    """
    decoded = []
    for e in range(q.shape[0]):
        n, c = sf[e].shape
        linear = sf[e].view(torch.uint8).reshape(n // 128, c // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(n, c)
        rows = torch.arange(n, device=q.device)
        if gated:
            rows = torch.where(rows < n // 2, rows * 2, (rows - n // 2) * 2 + 1)
        source = rows // 32 * 32 + rows % 4 * 8 + rows % 32 // 4
        decoded.append(decode(q[e][source], linear[source], 'mxfp4'))
    return torch.stack(decoded)

def expected(p):
    a = p['hidden_states'].float()
    w1 = weight(p['gemm1_weights'], p['gemm1_weights_scale'], True)
    w2 = weight(p['gemm2_weights'], p['gemm2_weights_scale'], False)
    ids, scores = p['topk_ids']
    y = torch.zeros_like(a)
    for e in range(w1.shape[0]):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        for off in range(0, len(tok), 64):
            rows = tok[off:off + 64]
            slots = slot[off:off + 64]
            up, gate = torch.einsum('th,ih->ti', a[rows], w1[e]).chunk(2, -1)
            alpha, beta = (p['gemm1_alpha'][e], p['gemm1_beta'][e])
            limit = p['gemm1_clamp_limit']
            if limit is not None:
                up = up.clamp(-limit[e], limit[e])
                gate = gate.clamp(max=limit[e])
            if p['activation_type'] == 10:
                act = alpha * torch.tanh(gate / alpha) * torch.sigmoid(gate) * (beta * torch.tanh(up / beta))
            else:
                act = gate * torch.sigmoid(alpha * gate) * (up + beta)
            y.index_add_(0, rows, torch.einsum('ti,hi->th', act, w2[e]) * scores[rows, slots, None])
    return y.to(p['output'].dtype)

class Model(torch.nn.Module):

    def forward(self, topk_ids, routing_bias, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta, gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, gemm2_bias, output1_scale_scalar, output1_scale_gate_scalar, output2_scale_scalar, num_experts, top_k, n_group, topk_group, intermediate_size, local_expert_offset, local_num_experts, routed_scaling_factor, routing_method_type=0, do_finalize=True, enable_pdl=None, activation_type=3, per_token_scale=None, output=None, tune_max_num_tokens=8192, gemm1_lora_delta=None, valid_hidden_size=None, valid_intermediate_size=None, num_fused_shared_experts=None, hidden_states_scale_layout=None):
        torch.backends.cuda.matmul.allow_tf32 = False
        p = {'topk_ids': topk_ids, 'routing_bias': routing_bias, 'hidden_states': hidden_states, 'hidden_states_scale': hidden_states_scale, 'gemm1_weights': gemm1_weights, 'gemm1_weights_scale': gemm1_weights_scale, 'gemm1_bias': gemm1_bias, 'gemm1_alpha': gemm1_alpha, 'gemm1_beta': gemm1_beta, 'gemm1_clamp_limit': gemm1_clamp_limit, 'gemm2_weights': gemm2_weights, 'gemm2_weights_scale': gemm2_weights_scale, 'gemm2_bias': gemm2_bias, 'output1_scale_scalar': output1_scale_scalar, 'output1_scale_gate_scalar': output1_scale_gate_scalar, 'output2_scale_scalar': output2_scale_scalar, 'num_experts': num_experts, 'top_k': top_k, 'n_group': n_group, 'topk_group': topk_group, 'intermediate_size': intermediate_size, 'local_expert_offset': local_expert_offset, 'local_num_experts': local_num_experts, 'routed_scaling_factor': routed_scaling_factor, 'routing_method_type': routing_method_type, 'do_finalize': do_finalize, 'enable_pdl': enable_pdl, 'activation_type': activation_type, 'per_token_scale': per_token_scale, 'output': output, 'tune_max_num_tokens': tune_max_num_tokens, 'gemm1_lora_delta': gemm1_lora_delta, 'valid_hidden_size': valid_hidden_size, 'valid_intermediate_size': valid_intermediate_size, 'num_fused_shared_experts': num_fused_shared_experts, 'hidden_states_scale_layout': hidden_states_scale_layout}
        result = expected(p)
        output.copy_(result)
        return [output]

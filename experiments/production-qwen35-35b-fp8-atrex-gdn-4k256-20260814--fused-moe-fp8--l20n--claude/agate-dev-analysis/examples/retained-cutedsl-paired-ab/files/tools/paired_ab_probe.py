"""Paired A/B gate for fused-MoE candidates: bitwise equality + interleaved timing.

Reusable harness used before spending an official Evaluate. It imports TWO in-pod
modules side by side:
  - `kernel`            : the candidate (work/kernel/kernel.py is uploaded under this name)
  - `kernel_incumbent`  : the incumbent to beat (upload input/kernel/kernel.py under THIS
                          base name via the dev request `file_paths`, so it does not shadow
                          the candidate's `kernel` base name)

For each T it (1) checks torch.equal bitwise equality on identical random inputs, then
(2) times both with interleaved alternating-order cuda-event pairs and reports medians.
Prints one JSON object per line (device line first, then one line per T).

Edit the T list / N_PAIRS / distributions in main()/make_inputs() to match the operator
contract. This version is written for the fused_moe_fp8 contract (hidden 2048, 256 experts,
top-k 8, e4m3 weights with f32 128x128/128-K scales).
"""
import json
import statistics

import torch

import kernel as cand_mod
import kernel_incumbent as inc_mod

TOPK = 8
NUM_EXPERTS = 256
HIDDEN = 2048
N_WARM = 5
N_PAIRS = 40
TS = (450, 512, 4096, 8192)


def make_inputs(T, gen):
    hidden = torch.randn(T, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=gen)
    # distinct top-k experts per token (realistic routing)
    ids = torch.rand(T, NUM_EXPERTS, device="cuda", generator=gen).topk(TOPK, dim=1).indices.to(torch.int32)
    w = torch.rand(T, TOPK, device="cuda", generator=gen).float()
    w = w / w.sum(dim=1, keepdim=True)
    w1 = (torch.randn(NUM_EXPERTS, 1024, HIDDEN, device="cuda", generator=gen) * 0.02).to(torch.float8_e4m3fn)
    w2 = (torch.randn(NUM_EXPERTS, HIDDEN, 512, device="cuda", generator=gen) * 0.02).to(torch.float8_e4m3fn)
    w1s = torch.rand(NUM_EXPERTS, 8, 16, device="cuda", generator=gen).float() + 0.5
    w2s = torch.rand(NUM_EXPERTS, 16, 4, device="cuda", generator=gen).float() + 0.5
    return hidden, w1, w2, w, ids, w1s, w2s


def main():
    torch.cuda.init()
    print(json.dumps({"device": torch.cuda.get_device_name(0)}), flush=True)
    cand = cand_mod.Model()
    inc = inc_mod.Model()

    for T in TS:
        gen = torch.Generator(device="cuda").manual_seed(1234 + T)
        args = make_inputs(T, gen)

        torch.cuda.synchronize()
        out_c = cand(*args)
        out_i = inc(*args)
        torch.cuda.synchronize()
        equal = torch.equal(out_c, out_i)
        maxdiff = (out_c.float() - out_i.float()).abs().max().item()

        for _ in range(N_WARM):
            cand(*args)
            inc(*args)
        torch.cuda.synchronize()

        ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        t_c, t_i = [], []
        for rep in range(N_PAIRS):
            if rep % 2 == 0:
                ev[0].record(); cand(*args); ev[1].record(); inc(*args); ev[2].record()
                torch.cuda.synchronize()
                t_c.append(ev[0].elapsed_time(ev[1])); t_i.append(ev[1].elapsed_time(ev[2]))
            else:
                ev[0].record(); inc(*args); ev[1].record(); cand(*args); ev[2].record()
                torch.cuda.synchronize()
                t_i.append(ev[0].elapsed_time(ev[1])); t_c.append(ev[1].elapsed_time(ev[2]))
        mc, mi = statistics.median(t_c), statistics.median(t_i)
        print(json.dumps({
            "T": T, "equal": bool(equal), "max_abs_diff": maxdiff,
            "cand_med_us": mc * 1e3, "inc_med_us": mi * 1e3,
            "speedup": round(mi / mc, 4),
        }), flush=True)
        del args, out_c, out_i
        torch.cuda.empty_cache()


main()

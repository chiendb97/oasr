#!/usr/bin/env python3
"""Migration parity gate: in-tree GPU WFST decoder vs the external ``_wfst.so``.

Decodes identical GPU log-prob batches on the same graph through both the in-tree
``oasr._C.decoder`` and the standalone ``wfst`` project's ``_wfst`` extension, and asserts
that every lane's ``words``, ``score`` (within ``--score-atol``) and ``overflow`` match.
The in-tree decoder is the same CUDA source compiled with the same flags and SM arch, so
results should be bit-for-bit identical — this script guards the migration against drift
and is the concrete evidence for the "output parity vs the original wfst" acceptance
criterion.

Requires: oasr built with ``OASR_USE_WFST_DECODER=1`` and the external ``wfst`` repo built
(``--wfst-home``, default ``/data01/kilm/users/chiendb/projects/wfst``) with its
``data/`` assets present.

Example::

    CUDA_VISIBLE_DEVICES=GPU-... python scripts/wfst_parity_check.py \
        --graph .../data/hlg/icefall.img \
        --logprobs .../data/logprobs/icefall-ljs2000.pt \
        --num-utts 256 --batch 1 8 32
"""

import argparse
import sys
from pathlib import Path

import torch

DEFAULT_WFST_HOME = Path("/data01/kilm/users/chiendb/projects/wfst")


def load_dump(path: str, limit):
    d = torch.load(path, map_location="cpu", weights_only=False)
    lps = d["log_probs"]
    order = sorted(range(len(lps)), key=lambda i: -lps[i].size(0))
    if limit:
        order = order[:limit]
    return [lps[i] for i in order]


def make_batches(log_probs, batch_size, device):
    out = []
    for i in range(0, len(log_probs), batch_size):
        lps = log_probs[i : i + batch_size]
        t_max = max(lp.size(0) for lp in lps)
        batch = torch.full((len(lps), t_max, lps[0].size(1)), -30.0)
        for k, lp in enumerate(lps):
            batch[k, : lp.size(0)] = lp
        lengths = torch.tensor([lp.size(0) for lp in lps], dtype=torch.int32)
        out.append((batch.to(device), lengths))
    return out


def make_decoder(lib, graph_path, args, device, t_max, max_lanes):
    graph = lib.load_graph(graph_path)
    dec = lib.GpuDecoder(
        graph,
        search_beam=args.search_beam,
        output_beam=args.output_beam,
        min_active=args.min_active,
        max_active=args.max_active,
        allow_partial=True,
        max_lanes=max_lanes,
        max_frames=t_max,
        device=device.index,
        main_q_factor=32,
        cand_factor=3,
    )
    return graph, dec  # keep graph referenced (decoder borrows it)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", default=str(DEFAULT_WFST_HOME / "data/hlg/primary.img"))
    p.add_argument("--logprobs", default=str(DEFAULT_WFST_HOME / "data/logprobs/primary-ljs2000.pt"))
    p.add_argument("--wfst-home", default=str(DEFAULT_WFST_HOME))
    p.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--num-utts", type=int, default=256)
    p.add_argument("--search-beam", type=float, default=20.0)
    p.add_argument("--output-beam", type=float, default=8.0)
    p.add_argument("--min-active", type=int, default=30)
    p.add_argument("--max-active", type=int, default=10000)
    p.add_argument("--score-atol", type=float, default=1e-4)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    import oasr._C as _C

    if not getattr(_C.decoder, "wfst_decoder_available", False):
        raise SystemExit("oasr was not built with OASR_USE_WFST_DECODER=1 (in-tree decoder missing)")
    intree = _C.decoder

    py = str(Path(args.wfst_home) / "python")
    if py not in sys.path:
        sys.path.insert(0, py)
    from wfst_decoder import get_lib

    external = get_lib()

    log_probs = load_dump(args.logprobs, args.num_utts)
    t_max = max(lp.size(0) for lp in log_probs)
    max_lanes = max(args.batch)

    _g_in, dec_in = make_decoder(intree, args.graph, args, device, t_max, max_lanes)
    _g_ex, dec_ex = make_decoder(external, args.graph, args, device, t_max, max_lanes)

    total = mism = 0
    for b in args.batch:
        b_mism = 0
        for batch, lengths in make_batches(log_probs, b, device):
            oi = dec_in.decode_batch(batch, lengths)
            oe = dec_ex.decode_batch(batch, lengths)
            for a, c in zip(oi, oe):
                total += 1
                w_ok = list(a["words"]) == list(c["words"])
                s_ok = abs(a["score"] - c["score"]) <= args.score_atol
                o_ok = a["overflow"] == c["overflow"]
                if not (w_ok and s_ok and o_ok):
                    mism += 1
                    b_mism += 1
                    if mism <= 10:
                        print(
                            f"  MISMATCH B={b}: words_ok={w_ok} "
                            f"score={a['score']:.5f}/{c['score']:.5f} "
                            f"overflow={a['overflow']}/{c['overflow']}"
                        )
        print(f"B={b}: {b_mism} mismatch(es)")

    ok = total - mism
    print(f"\n{ok}/{total} lanes identical (words + score + overflow).")
    if mism:
        raise SystemExit(f"PARITY FAILED: {mism} mismatch(es)")
    print("PARITY OK: in-tree decoder == external _wfst on every lane.")


if __name__ == "__main__":
    main()

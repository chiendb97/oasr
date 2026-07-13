#!/usr/bin/env python3
"""Migration parity gate: in-tree GPU WFST decoder vs the external ``_wfst.so``.

Decodes identical GPU log-prob batches on the same graph through both the in-tree
JIT-compiled decoder (``oasr.jit.wfst_decoder``, driven over TVM-FFI via int64 handles)
and the standalone ``wfst`` project's pybind ``_wfst`` extension, and asserts that every
lane's ``words``, ``score`` (within ``--score-atol``) and ``overflow`` match.  The in-tree
decoder is the same CUDA source compiled with the same flags/SM arch, so results are
bit-for-bit identical — this guards the migration against drift and is the concrete
evidence for the "output parity vs the original wfst" acceptance criterion.

Requires: a CUDA build of oasr (the decoder JIT-compiles on first use) and the external
``wfst`` repo built (``--wfst-home``, default ``/data01/kilm/users/chiendb/projects/wfst``)
with its ``data/`` assets present.

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


class IntreeDecoder:
    """In-tree JIT decoder driven over TVM-FFI int64 handles."""

    def __init__(self, args, device, t_max, max_lanes):
        from oasr.jit.wfst_decoder import gen_wfst_decoder_module

        self.mod = gen_wfst_decoder_module().build_and_load()
        self.dev = device
        gh = int(self.mod.wfst_load_graph(args.graph))
        self.dec = int(self.mod.wfst_create_decoder(
            gh, args.search_beam, args.output_beam, args.min_active, args.max_active, 1,
            max_lanes, t_max, device.index, 32, 3, 1, 0, 0, 0, 0, 3, 0, 0,
            args.gc_interval))

    def decode(self, batch, lengths):
        g = int(batch.size(0))
        lens_t = lengths.to(torch.int32).cpu()
        cap = max(1, int(batch.size(1)))
        while True:
            ow = torch.empty((g, cap), dtype=torch.int32)
            wl = torch.empty((g,), dtype=torch.int32)
            sc = torch.empty((g,), dtype=torch.float64)
            meta = torch.empty((g, 3), dtype=torch.int32)
            self.mod.wfst_decode_batch(self.dec, batch.contiguous(), lens_t, ow, wl, sc, meta)
            if g == 0 or int(wl.max()) <= cap:
                break
            cap = int(wl.max())
        return [
            (ow[b, : min(int(wl[b]), cap)].tolist(), float(sc[b]), int(meta[b, 2]))
            for b in range(g)
        ]


class ExternalDecoder:
    """External standalone-wfst pybind decoder (``_wfst.so``)."""

    def __init__(self, args, device, t_max, max_lanes):
        py = str(Path(args.wfst_home) / "python")
        if py not in sys.path:
            sys.path.insert(0, py)
        from wfst_decoder import get_lib

        lib = get_lib()
        self.graph = lib.load_graph(args.graph)  # keep referenced (decoder borrows it)
        self.dec = lib.GpuDecoder(
            self.graph, search_beam=args.search_beam, output_beam=args.output_beam,
            min_active=args.min_active, max_active=args.max_active, allow_partial=True,
            max_lanes=max_lanes, max_frames=t_max, device=device.index,
            main_q_factor=32, cand_factor=3)

    def decode(self, batch, lengths):
        outs = self.dec.decode_batch(batch.contiguous(), lengths.to(torch.int32).cpu())
        return [(list(o["words"]), float(o["score"]), int(o["overflow"])) for o in outs]


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
    p.add_argument("--gc-interval", type=int, default=0,
                   help="in-tree winners-log GC cadence (0 = off); results must be "
                        "identical either way — parity with GC on proves GC exactness")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    log_probs = load_dump(args.logprobs, args.num_utts)
    t_max = max(lp.size(0) for lp in log_probs)
    max_lanes = max(args.batch)

    intree = IntreeDecoder(args, device, t_max, max_lanes)
    external = ExternalDecoder(args, device, t_max, max_lanes)

    total = mism = 0
    for b in args.batch:
        b_mism = 0
        for batch, lengths in make_batches(log_probs, b, device):
            oi = intree.decode(batch, lengths)
            oe = external.decode(batch, lengths)
            for (wi, si, fi), (we, se, fe) in zip(oi, oe):
                total += 1
                if not (wi == we and abs(si - se) <= args.score_atol and fi == fe):
                    mism += 1
                    b_mism += 1
                    if mism <= 10:
                        print(f"  MISMATCH B={b}: words_ok={wi == we} "
                              f"score={si:.5f}/{se:.5f} overflow={fi}/{fe}")
        print(f"B={b}: {b_mism} mismatch(es)")

    ok = total - mism
    print(f"\n{ok}/{total} lanes identical (words + score + overflow).")
    if mism:
        raise SystemExit(f"PARITY FAILED: {mism} mismatch(es)")
    print("PARITY OK: in-tree JIT decoder == external _wfst on every lane.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark WFST decoding from device-resident log probabilities to words.

The timing includes read-back, host backtracking, and word mapping. Assets may
be passed directly or resolved below ``$OASR_WFST_HOME``.
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch

#: Root of the external ``wfst`` checkout.  No baked-in default — the assets
#: live outside this repo and their location is per-machine.
WFST_HOME_ENV = "OASR_WFST_HOME"


def _wfst_asset(*parts: str):
    """Default path under ``$OASR_WFST_HOME``, or ``None`` when it is unset."""
    home = os.environ.get(WFST_HOME_ENV)
    return str(Path(home, *parts)) if home else None


def load_dump(path: str):
    d = torch.load(path, map_location="cpu", weights_only=False)
    lps = d["log_probs"]
    order = sorted(range(len(lps)), key=lambda i: -lps[i].size(0))
    return [lps[i] for i in order]


def make_gpu_batches(log_probs, batch_size, device, frame_sec):
    """Length-sorted padded GPU batches: list of (tensor [B,T,V], lengths, audio_s)."""
    out = []
    for i in range(0, len(log_probs), batch_size):
        lps = log_probs[i : i + batch_size]
        t_max = max(lp.size(0) for lp in lps)
        batch = torch.full((len(lps), t_max, lps[0].size(1)), -30.0)
        for k, lp in enumerate(lps):
            batch[k, : lp.size(0)] = lp
        lengths = torch.tensor([lp.size(0) for lp in lps], dtype=torch.int32)
        out.append((batch.to(device), lengths, sum(lp.size(0) for lp in lps) * frame_sec))
    return out


class IntreeEngine:
    """In-tree JIT decoder (TVM-FFI int64 handles) with overflow-rescue decode."""

    name = "in-tree"

    def __init__(self, args, device, t_max):
        from oasr.jit.wfst_decoder import gen_wfst_decoder_module

        self.mod = gen_wfst_decoder_module().build_and_load()
        self.args = args
        self.device = device
        self.t_max = t_max
        self.graph = int(self.mod.wfst_load_graph(args.graph))
        self.dec = self._make(max_lanes=args.max_batch, main_q_factor=32)
        self._rescue = None
        self.overflows = 0

    def _make(self, max_lanes, main_q_factor):
        return int(
            self.mod.wfst_create_decoder(
                self.graph,
                self.args.search_beam,
                self.args.output_beam,
                self.args.min_active,
                self.args.max_active,
                1,  # allow_partial
                max_lanes,
                self.t_max,
                self.device.index,
                main_q_factor,
                3,
                1,
                0,
                0,
                0,
                0,
                3,
                0,
                0,
                0,
            )
        )

    def _decode_into(self, dec, batch, lens_t):
        g = int(batch.size(0))
        cap = max(1, int(batch.size(1)))
        while True:
            ow = torch.empty((g, cap), dtype=torch.int32)
            wl = torch.empty((g,), dtype=torch.int32)
            sc = torch.empty((g,), dtype=torch.float64)
            meta = torch.empty((g, 3), dtype=torch.int32)
            self.mod.wfst_decode_batch(dec, batch.contiguous(), lens_t, ow, wl, sc, meta)
            if g == 0 or int(wl.max()) <= cap:
                break
            cap = int(wl.max())
        return [(ow[i, : min(int(wl[i]), cap)].tolist(), int(meta[i, 2])) for i in range(g)]

    def decode(self, batch, lengths):
        lens_t = lengths.to(torch.int32).cpu()
        res = self._decode_into(self.dec, batch, lens_t)
        bad = [i for i, (_, o) in enumerate(res) if o != 0]
        if bad:
            self.overflows += len(bad)
            if self._rescue is None:
                self._rescue = self._make(max_lanes=2, main_q_factor=64)
            for i in range(0, len(bad), 2):
                idx = bad[i : i + 2]
                sub = batch[idx].contiguous()
                sub_len = lens_t[torch.tensor(idx)]
                for j, out in enumerate(self._decode_into(self._rescue, sub, sub_len)):
                    res[idx[j]] = out
        return [w for w, _ in res]


class ExternalEngine:
    """External standalone-wfst pybind decoder (``_wfst.so``) with overflow rescue."""

    name = "external"

    def __init__(self, args, device, t_max):
        py = str(Path(args.wfst_home) / "python")
        if py not in sys.path:
            sys.path.insert(0, py)
        from wfst_decoder import get_lib

        self.lib = get_lib()
        self.args = args
        self.device = device
        self.t_max = t_max
        self.graph = self.lib.load_graph(args.graph)
        self.dec = self._make(max_lanes=args.max_batch, main_q_factor=32)
        self._rescue = None
        self.overflows = 0

    def _make(self, max_lanes, main_q_factor):
        return self.lib.GpuDecoder(
            self.graph,
            search_beam=self.args.search_beam,
            output_beam=self.args.output_beam,
            min_active=self.args.min_active,
            max_active=self.args.max_active,
            allow_partial=True,
            max_lanes=max_lanes,
            max_frames=self.t_max,
            device=self.device.index,
            main_q_factor=main_q_factor,
            cand_factor=3,
        )

    def decode(self, batch, lengths):
        outs = self.dec.decode_batch(batch.contiguous(), lengths.to(torch.int32).cpu())
        bad = [i for i, o in enumerate(outs) if o["overflow"] != 0]
        if bad:
            self.overflows += len(bad)
            if self._rescue is None:
                self._rescue = self._make(max_lanes=2, main_q_factor=64)
            for i in range(0, len(bad), 2):
                idx = bad[i : i + 2]
                sub = batch[idx].contiguous()
                sub_len = lengths[torch.tensor(idx)]
                for k, o in zip(idx, self._rescue.decode_batch(sub, sub_len)):
                    outs[k] = o
        return [list(o["words"]) for o in outs]


def run_time(engine, gpu_batches, args):
    sel = gpu_batches[: args.num_batches]
    for batch, lengths, _ in sel[:3]:  # warmup (CUDA-graph capture)
        engine.decode(batch, lengths)
    torch.cuda.synchronize()
    per = []
    for _ in range(args.repeats):
        for batch, lengths, audio_s in sel:
            t0 = time.perf_counter()
            engine.decode(batch, lengths)
            torch.cuda.synchronize()
            per.append((time.perf_counter() - t0, audio_s))
    times = sorted(t for t, _ in per)
    inv = statistics.median(a / t for t, a in per)
    print(
        f"[{engine.name}] median {statistics.median(times) * 1e3:.1f} ms, "
        f"p95 {times[int(0.95 * len(times)) - 1] * 1e3:.1f} ms, inv-RTF median {inv:.0f}x"
    )
    return inv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--graph",
        default=_wfst_asset("data/hlg/primary.img"),
        help=f"decoding graph image (default: ${WFST_HOME_ENV}/data/hlg/primary.img)",
    )
    p.add_argument(
        "--logprobs",
        default=_wfst_asset("data/logprobs/primary-ljs2000.pt"),
        help=f"log-prob dump (default: ${WFST_HOME_ENV}/data/logprobs/primary-ljs2000.pt)",
    )
    p.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--search-beam", type=float, default=20.0)
    p.add_argument("--output-beam", type=float, default=8.0)
    p.add_argument("--min-active", type=int, default=30)
    p.add_argument("--max-active", type=int, default=10000)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--num-batches", type=int, default=6)
    p.add_argument("--frame-sec", type=float, default=0.04, help="seconds per frame (RTF scale)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--compare-external",
        action="store_true",
        help="also time the external _wfst.so build (perf A/B)",
    )
    p.add_argument(
        "--wfst-home",
        default=os.environ.get(WFST_HOME_ENV),
        help=f"external wfst checkout, holding python/ (default: ${WFST_HOME_ENV}); "
        "only needed with --compare-external",
    )
    args = p.parse_args()
    args.max_batch = max(args.batch)

    missing = [
        flag
        for flag, value in (("--graph", args.graph), ("--logprobs", args.logprobs))
        if not value
    ]
    if args.compare_external and not args.wfst_home:
        missing.append("--wfst-home")
    if missing:
        raise SystemExit(
            f"missing {', '.join(missing)} — pass them, or set ${WFST_HOME_ENV} to the "
            f"external wfst checkout so they default to its data/ layout"
        )

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    log_probs = load_dump(args.logprobs)
    t_max = max(lp.size(0) for lp in log_probs)

    engines = [IntreeEngine(args, device, t_max)]
    if args.compare_external:
        engines.append(ExternalEngine(args, device, t_max))

    for b in args.batch:
        gpu_batches = make_gpu_batches(log_probs, b, device, args.frame_sec)
        print(
            f"=== B={b} beam={args.search_beam}/{args.output_beam} max_active={args.max_active} ==="
        )
        for engine in engines:
            run_time(engine, gpu_batches, args)
        del gpu_batches
        torch.cuda.empty_cache()

    for engine in engines:
        if engine.overflows:
            print(f"[{engine.name}] NOTE: {engine.overflows} lanes hit overflow (rescued exactly)")


if __name__ == "__main__":
    main()

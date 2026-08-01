"""Binary graph-image format (hlg.img) — canonical definition.

Little-endian, 128-byte-aligned sections, fixed 256-byte header:

    magic            u64   0x31474D4954534657 ("WFSTIMG1" little-endian)
    version          u32   2 (v1 files remain readable: no epsilon section)
    flags            u32   bit0 = finals_at_end (final arcs last within a state's range)
                           bit1 = has_epsilons
                           bit2 = eps_first (epsilon arcs at the START of the non-final
                                  range; else at its end)
    num_states       i64
    num_arcs         i64
    vocab_size       i32   (= max ilabel + 1; ilabel -1 marks final arcs)
    start_state      i32   (k2 convention: 0)
    aux_pool_size    i64
    section offsets  7 x i64 (bytes from file start; v1 headers carry 6):
        row_splits, final_count, arc_dest_ilabel, arc_weight, aux_row_splits, aux_pool,
        eps_count

Sections:
    row_splits       i32[num_states + 1]   CSR arc offsets (k2 arc order preserved)
    final_count      i32[num_states]       #ilabel==-1 arcs in the state's range
    arc_dest_ilabel  i32[2 * num_arcs]     interleaved {dest, ilabel} (int2 loads on GPU)
    arc_weight       f32[num_arcs]
    aux_row_splits   i32[num_arcs + 1]     ragged aux labels (word ids) CSR
    aux_pool         i32[aux_pool_size]
    eps_count        i32[num_states]       #epsilon (non-emitting) arcs per state (v2)

Epsilon arcs (OpenFST-style TLG graphs) consume no frame; the decoder identifies them by
segment, never by their stored ilabel (which is preserved verbatim for arc_map fidelity —
note that in k2 HLG graphs label 0 means BLANK, an emitting label, so epsilon-ness is an
export-time declaration via epsilon_id, not a label convention).

Arc order is IDENTICAL to the source k2 FSA: arc index == k2 graph arc index, so lattice
arc_map_a is the identity mapping (exact aux_labels parity). Final-arc placement within a
state (start vs end) is detected, asserted contiguous, and recorded in flags.
"""

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = 0x31474D4954534657
VERSION = 2
FLAG_FINALS_AT_END = 1
FLAG_HAS_EPS = 2
FLAG_EPS_FIRST = 4
_HEADER_BYTES = 256
_ALIGN = 128
_HDR_V1 = "<QIIqqiiq6q"
_HDR_V2 = "<QIIqqiiq7q"


def _align(off: int) -> int:
    return (off + _ALIGN - 1) // _ALIGN * _ALIGN


@dataclass
class GraphImage:
    row_splits: np.ndarray  # i32 [N+1]
    final_count: np.ndarray  # i32 [N]
    arc_dest_ilabel: np.ndarray  # i32 [2A] interleaved
    arc_weight: np.ndarray  # f32 [A]
    aux_row_splits: np.ndarray  # i32 [A+1]
    aux_pool: np.ndarray  # i32
    vocab_size: int
    finals_at_end: bool
    start_state: int = 0
    eps_count: np.ndarray | None = None  # i32 [N]; None == epsilon-free
    eps_first: bool = False  # eps arcs at the start of the non-final range

    @property
    def num_states(self) -> int:
        return len(self.row_splits) - 1

    @property
    def num_arcs(self) -> int:
        return len(self.arc_weight)


def write_image(img: GraphImage, path: str | Path) -> None:
    has_eps = img.eps_count is not None and int(np.sum(img.eps_count)) > 0
    eps_count = (
        np.ascontiguousarray(img.eps_count, dtype=np.int32)
        if img.eps_count is not None
        else np.zeros(img.num_states, dtype=np.int32)
    )
    sections = [
        np.ascontiguousarray(img.row_splits, dtype=np.int32),
        np.ascontiguousarray(img.final_count, dtype=np.int32),
        np.ascontiguousarray(img.arc_dest_ilabel, dtype=np.int32),
        np.ascontiguousarray(img.arc_weight, dtype=np.float32),
        np.ascontiguousarray(img.aux_row_splits, dtype=np.int32),
        np.ascontiguousarray(img.aux_pool, dtype=np.int32),
        eps_count,
    ]
    assert len(sections[0]) == img.num_states + 1
    assert len(sections[1]) == img.num_states
    assert len(sections[2]) == 2 * img.num_arcs
    assert len(sections[4]) == img.num_arcs + 1

    offsets = []
    off = _HEADER_BYTES
    for s in sections:
        off = _align(off)
        offsets.append(off)
        off += s.nbytes

    flags = (
        (FLAG_FINALS_AT_END if img.finals_at_end else 0)
        | (FLAG_HAS_EPS if has_eps else 0)
        | (FLAG_EPS_FIRST if img.eps_first else 0)
    )
    header = struct.pack(
        _HDR_V2,
        MAGIC,
        VERSION,
        flags,
        img.num_states,
        img.num_arcs,
        img.vocab_size,
        img.start_state,
        len(img.aux_pool),
        *offsets,
    )
    header += b"\0" * (_HEADER_BYTES - len(header))

    with open(path, "wb") as f:
        f.write(header)
        for s, o in zip(sections, offsets):
            f.seek(o)
            f.write(s.tobytes())


def read_image(path: str | Path) -> GraphImage:
    with open(path, "rb") as f:
        header = f.read(_HEADER_BYTES)
        magic, version = struct.unpack("<QI", header[:12])
        if magic != MAGIC:
            raise ValueError(f"bad magic {magic:#x} in {path}")
        if version not in (1, 2):
            raise ValueError(f"unsupported version {version}")
        fmt = _HDR_V2 if version == 2 else _HDR_V1
        _, _, flags, n_states, n_arcs, vocab, start, aux_size, *offs = struct.unpack(
            fmt, header[: struct.calcsize(fmt)]
        )

        def rd(off: int, count: int, dtype) -> np.ndarray:
            f.seek(off)
            return np.frombuffer(f.read(count * np.dtype(dtype).itemsize), dtype=dtype).copy()

        eps_count = rd(offs[6], n_states, np.int32) if version == 2 else None
        return GraphImage(
            row_splits=rd(offs[0], n_states + 1, np.int32),
            final_count=rd(offs[1], n_states, np.int32),
            arc_dest_ilabel=rd(offs[2], 2 * n_arcs, np.int32),
            arc_weight=rd(offs[3], n_arcs, np.float32),
            aux_row_splits=rd(offs[4], n_arcs + 1, np.int32),
            aux_pool=rd(offs[5], aux_size, np.int32),
            vocab_size=vocab,
            finals_at_end=bool(flags & FLAG_FINALS_AT_END),
            start_state=start,
            eps_count=eps_count,
            eps_first=bool(flags & FLAG_EPS_FIRST),
        )


def build_image(
    row_splits: np.ndarray,
    dest: np.ndarray,
    ilabel: np.ndarray,
    weight: np.ndarray,
    aux_row_splits: np.ndarray | None = None,
    aux_pool: np.ndarray | None = None,
    vocab_size: int | None = None,
    epsilon_id: int | None = None,
) -> GraphImage:
    """Assemble + validate an image from flat arrays (used by the exporter and by tests).

    Detects final-arc placement, asserts per-state contiguity, checks label ranges.
    """
    row_splits = np.asarray(row_splits, dtype=np.int32)
    dest = np.asarray(dest, dtype=np.int32)
    ilabel = np.asarray(ilabel, dtype=np.int32)
    weight = np.asarray(weight, dtype=np.float32)
    n_states = len(row_splits) - 1
    n_arcs = len(ilabel)
    assert row_splits[0] == 0 and row_splits[-1] == n_arcs
    assert len(dest) == n_arcs and len(weight) == n_arcs
    assert n_states < 2**31, "state ids must fit int32"

    if vocab_size is None:
        vocab_size = int(ilabel.max()) + 1 if n_arcs else 1
    if ilabel.size:
        assert ilabel.min() >= -1, "labels must be >= -1"
        assert ilabel.max() < vocab_size, (ilabel.max(), vocab_size)
        assert (dest >= 0).all() and (dest < n_states).all(), "dest out of range"

    deg = row_splits[1:] - row_splits[:-1]
    src = np.repeat(np.arange(n_states, dtype=np.int64), deg)
    is_final = ilabel == -1
    final_count = np.zeros(n_states, dtype=np.int64)
    np.add.at(final_count, src[is_final], 1)

    # Contiguity + placement: finals must occupy either the first or the last
    # final_count[s] slots of every state's range.
    pos = np.arange(n_arcs, dtype=np.int64) - row_splits[:-1].astype(np.int64).repeat(deg)
    finals_at_start = (
        bool((pos[is_final] < final_count[src[is_final]]).all()) if is_final.any() else False
    )
    finals_at_end = (
        bool((pos[is_final] >= (deg.astype(np.int64) - final_count)[src[is_final]]).all())
        if is_final.any()
        else True
    )
    if not (finals_at_start or finals_at_end):
        raise ValueError(
            "final arcs are not contiguous at one end of their state's range; "
            "graph needs a stable partition (unsupported layout)"
        )

    # Epsilon segment: declared by id (never inferred — k2 HLG label 0 is BLANK), must be
    # contiguous at one end of the state's non-final range.
    eps_count = None
    eps_first = False
    if epsilon_id is not None:
        is_eps = (ilabel == epsilon_id) & ~is_final
        eps_count = np.zeros(n_states, dtype=np.int64)
        np.add.at(eps_count, src[is_eps], 1)
        if is_eps.any():
            f_at_end = finals_at_end and not finals_at_start
            rest_begin = (row_splits[:-1] + (0 if f_at_end else 1) * final_count).astype(np.int64)
            rest_len = deg.astype(np.int64) - final_count
            pos_rest = np.arange(n_arcs, dtype=np.int64) - rest_begin.repeat(deg)
            ep, es = pos_rest[is_eps], src[is_eps]
            eps_at_start = bool((ep < eps_count[es]).all())
            eps_at_end = bool((ep >= (rest_len - eps_count)[es]).all())
            if not (eps_at_start or eps_at_end):
                raise ValueError(
                    "epsilon arcs are not contiguous at one end of the non-final range"
                )
            eps_first = eps_at_start and not eps_at_end

    if aux_row_splits is None:
        aux_row_splits = np.zeros(n_arcs + 1, dtype=np.int32)
        aux_pool = np.zeros(0, dtype=np.int32)
    aux_row_splits = np.asarray(aux_row_splits, dtype=np.int32)
    aux_pool = np.asarray(aux_pool, dtype=np.int32)
    assert len(aux_row_splits) == n_arcs + 1

    interleaved = np.empty(2 * n_arcs, dtype=np.int32)
    interleaved[0::2] = dest
    interleaved[1::2] = ilabel

    return GraphImage(
        row_splits=row_splits,
        final_count=final_count.astype(np.int32),
        arc_dest_ilabel=interleaved,
        arc_weight=weight,
        aux_row_splits=aux_row_splits,
        aux_pool=aux_pool,
        vocab_size=int(vocab_size),
        finals_at_end=finals_at_end and not finals_at_start,
        eps_count=eps_count.astype(np.int32) if eps_count is not None else None,
        eps_first=eps_first,
    )

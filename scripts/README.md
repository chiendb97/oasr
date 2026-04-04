# HLG Graph Building Scripts

Scripts for building HLG decoding graphs for CTC WFST decoding.
Adapted from the [Icefall AISHELL recipe](https://github.com/k2-fsa/icefall/tree/master/egs/aishell/ASR).

**HLG = H ∘ L ∘ G**

| Transducer | Description |
|------------|-------------|
| **H** | CTC topology — maps CTC output sequences (with blank/repeat removal) to token sequences |
| **L** | Lexicon — maps token sequences to word sequences |
| **G** | N-gram language model — scores word sequences |

The compiled `HLG.pt` is loaded directly by the OASR WFST decoder:

```python
from oasr.decode import Decoder, DecoderConfig

decoder = Decoder(DecoderConfig(search_type="wfst"), fst="data/lang_char/HLG.pt")
result = decoder.decode(log_probs)   # log_probs: torch.Tensor [T, V]
```

---

## Installation

```bash
pip install "oasr[wfst]"   # installs k2 and kaldilm
```

---

## Scripts

| Script | Description |
|--------|-------------|
| `lang_utils.py` | Shared utilities (not run directly) |
| `prepare_char.py` | Character-level lexicon preparation |
| `prepare_lang.py` | Phone-level lexicon preparation |
| `make_kn_lm.py` | Kneser-Ney n-gram LM training |
| `compile_hlg.py` | HLG graph compilation |
| `compile_lg.py` | LG graph compilation (without H) |

---

## End-to-End Workflow

### Character-based (AISHELL style)

#### Step 1 — Prepare the lang directory

The lang directory must contain:
- `text` — training transcripts, one utterance per line (raw characters, no spaces required)
- `words.txt` — word symbol table (`word id` per line)

```bash
python scripts/prepare_char.py --lang-dir data/lang_char
```

Outputs written to `data/lang_char/`:
- `tokens.txt` — character token-to-ID mapping
- `lexicon.txt` — word-to-character mapping
- `lexicon_disambig.txt` — lexicon with disambiguation symbols
- `L.pt` — lexicon FST (k2 format)
- `L_disambig.pt` — lexicon FST with `#0` self-loops

#### Step 2 — Train an n-gram language model

```bash
python scripts/make_kn_lm.py \
    -ngram-order 3 \
    -text data/lang_char/text \
    -lm data/lm/G_3_gram.arpa
```

The `-text` file should contain one sentence per line with space-separated words.
The script implements unmodified Kneser-Ney smoothing with no external dependencies.

#### Step 3 — Convert ARPA to OpenFst text format

```bash
python -m kaldilm \
    --read-symbol-table="data/lang_char/words.txt" \
    --disambig-symbol='#0' \
    --max-order=3 \
    data/lm/G_3_gram.arpa > data/lm/G_3_gram.fst.txt
```

#### Step 4 — Compile HLG

```bash
python scripts/compile_hlg.py \
    --lang-dir data/lang_char \
    --lm G_3_gram \
    --lm-dir data/lm
```

Output: `data/lang_char/HLG.pt`

On subsequent runs the script skips compilation if `HLG.pt` already exists.
Use `--overwrite` to force recompilation.

---

### Phone-based

Replace step 1 with `prepare_lang.py`, which reads a `lexicon.txt` with phone pronunciations
(e.g., `hello HH AH L OW`) and models inter-word silence:

```bash
python scripts/prepare_lang.py \
    --lang-dir data/lang_phone \
    --sil-token SIL \
    --sil-prob 0.5
```

Steps 2–4 are identical.

---

## Directory Layout

```
data/
├── lang_char/
│   ├── text                  # input: training transcripts
│   ├── words.txt             # input: word symbol table
│   ├── tokens.txt            # output: token symbol table
│   ├── lexicon.txt           # output: word→char mapping
│   ├── lexicon_disambig.txt  # output: with disambiguation symbols
│   ├── L.pt                  # output: lexicon FST
│   ├── L_disambig.pt         # output: lexicon FST with self-loops
│   └── HLG.pt                # output: final decoding graph
└── lm/
    ├── G_3_gram.arpa         # output of make_kn_lm.py
    ├── G_3_gram.fst.txt      # output of kaldilm
    └── G_3_gram.pt           # cached compiled G (auto-generated)
```

---

## Script Reference

### `prepare_char.py`

```
usage: prepare_char.py --lang-dir DIR

arguments:
  --lang-dir    Input/output lang directory (must contain 'text' and 'words.txt')
```

### `prepare_lang.py`

```
usage: prepare_lang.py --lang-dir DIR [--sil-token TOKEN] [--sil-prob PROB]

arguments:
  --lang-dir    Input/output lang directory (must contain 'lexicon.txt')
  --sil-token   Silence phone token (default: SIL)
  --sil-prob    Silence insertion probability at word boundaries (default: 0.5)
```

### `make_kn_lm.py`

```
usage: make_kn_lm.py [-ngram-order N] [-text FILE] [-lm FILE]

arguments:
  -ngram-order  N-gram order, 1–7 (default: 3)
  -text         Input corpus file; reads stdin if omitted
  -lm           Output ARPA file; writes stdout if omitted
```

### `compile_hlg.py`

```
usage: compile_hlg.py --lang-dir DIR [--lm STEM] [--lm-dir DIR] [--overwrite]

arguments:
  --lang-dir    Lang directory with L_disambig.pt, tokens.txt, words.txt
  --lm          LM file stem name, e.g. G_3_gram (default: G_3_gram)
                Expects {lm_dir}/{lm}.fst.txt or {lm_dir}/{lm}.pt
  --lm-dir      Directory containing the LM file (default: data/lm)
  --overwrite   Recompile even if HLG.pt already exists
```

### `compile_lg.py`

```
usage: compile_lg.py --lang-dir DIR [--lm STEM] [--lm-dir DIR] [--overwrite]

Same arguments as compile_hlg.py. Outputs LG.pt instead of HLG.pt.
Use this when the CTC topology (H) is applied separately at decode time.
```

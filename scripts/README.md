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
| `run_hlg.sh` | End-to-end pipeline script (all token types, stage control) |
| `lang_utils.py` | Shared utilities (not run directly) |
| `prepare_char.py` | Character-level lexicon preparation |
| `prepare_lang.py` | Phone-level lexicon preparation |
| `prepare_words.py` | Build `words.txt` word symbol table from plain-text training data |
| `prepare_bpe.py` | BPE/subword lexicon preparation (SentencePiece) |
| `make_kn_lm.py` | Kneser-Ney n-gram LM training (accepts multiple text files) |
| `compile_hlg.py` | HLG graph compilation |
| `compile_lg.py` | LG graph compilation (without H) |

---

## End-to-End Pipeline Script

`run_hlg.sh` runs all stages in sequence with stage skipping and resume support.

```bash
# BPE (e.g. LibriSpeech WeNet/ESPnet model)
bash scripts/run_hlg.sh \
    --token-type bpe \
    --am-dir /path/to/am \
    --text   train-clean-100/text train-clean-360/text train-other-500/text \
    --lang-dir data/lang_bpe \
    --lm-dir   data/lm

# Character-based
bash scripts/run_hlg.sh \
    --token-type char \
    --lang-dir data/lang_char \
    --lm-dir   data/lm

# Phone-based
bash scripts/run_hlg.sh \
    --token-type phone \
    --lang-dir data/lang_phone \
    --lm-dir   data/lm \
    --text     corpus/text
```

**Resume from a stage** (outputs of earlier stages are reused):

```bash
bash scripts/run_hlg.sh ... --stage 3
```

**Run only one stage**:

```bash
bash scripts/run_hlg.sh ... --stage 4 --stop-stage 4
```

**Force recomputation** even if outputs already exist:

```bash
bash scripts/run_hlg.sh ... --overwrite
```

| Option | Default | Description |
|--------|---------|-------------|
| `--token-type` | `bpe` | `bpe`, `char`, or `phone` |
| `--am-dir` | — | Acoustic model directory (BPE: provides default `--bpe-model` and `--units-file`) |
| `--bpe-model` | `$am_dir/train_960_unigram5000.model` | SentencePiece model path |
| `--units-file` | `$am_dir/units.txt` | BPE token symbol table |
| `--text` | — | Plain-text corpus files (one sentence per line) |
| `--lang-dir` | `data/lang_bpe` | Lang directory |
| `--lm-dir` | `data/lm` | LM output directory |
| `--lm-stem` | `G_3_gram` | LM filename stem |
| `--ngram-order` | `3` | N-gram order |
| `--min-word-count` | `1` | Min word frequency for `words.txt` |
| `--sil-token` | `SIL` | Silence phone token (phone mode only) |
| `--sil-prob` | `0.5` | Silence insertion probability (phone mode only) |
| `--stage` | `1` | Start from this stage |
| `--stop-stage` | `5` | Stop after this stage |
| `--overwrite` | off | Recompute even if outputs exist |

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

Input text files must contain one sentence per line with space-separated words
(no utterance IDs).  Multiple files can be passed to `-text`.
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

### BPE / SentencePiece (LibriSpeech style)

Use this workflow for WeNet / ESPnet models trained with a SentencePiece BPE or
unigram vocabulary (e.g. `train_960_unigram5000.model`).

#### Step 1 — Build the word symbol table

```bash
python scripts/prepare_words.py \
    --text train-clean-100/text train-clean-360/text train-other-500/text \
    --lang-dir data/lang_bpe
```

Input text files must contain one sentence per line with space-separated words
(no utterance IDs).  Outputs `data/lang_bpe/words.txt`.

#### Step 2 — Prepare the BPE lang directory

```bash
AM=/path/to/20210610_u2pp_conformer_exp_librispeech

python scripts/prepare_bpe.py \
    --lang-dir data/lang_bpe \
    --bpe-model $AM/train_960_unigram5000.model \
    --units-file $AM/units.txt
```

Outputs written to `data/lang_bpe/`:
- `tokens.txt` — BPE token symbol table (from the AM `units.txt`, extended with disambiguation symbols)
- `lexicon.txt` — word → BPE-piece sequence mapping
- `lexicon_disambig.txt` — lexicon with disambiguation symbols
- `L.pt` — lexicon FST
- `L_disambig.pt` — lexicon FST with `#0` self-loops

#### Step 3 — Train an n-gram language model

```bash
python scripts/make_kn_lm.py \
    -ngram-order 3 \
    -text train-clean-100/text train-clean-360/text train-other-500/text \
    -lm data/lm/G_3_gram.arpa
```

#### Step 4 — Convert ARPA to OpenFst text format

```bash
python -m kaldilm \
    --read-symbol-table="data/lang_bpe/words.txt" \
    --disambig-symbol='#0' \
    --max-order=3 \
    data/lm/G_3_gram.arpa > data/lm/G_3_gram.fst.txt
```

#### Step 5 — Compile HLG

```bash
python scripts/compile_hlg.py \
    --lang-dir data/lang_bpe \
    --lm G_3_gram \
    --lm-dir data/lm
```

Output: `data/lang_bpe/HLG.pt`


**Decoding with the compiled graph:**

```python
from oasr.decode import Decoder, DecoderConfig

decoder = Decoder(
    DecoderConfig(search_type="wfst"),
    fst="data/lang_bpe/HLG.pt",
)
result = decoder.decode(log_probs)   # log_probs: torch.Tensor [T, V]
```

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
├── lang_char/                # character-based workflow outputs
│   ├── text                  # input: training transcripts
│   ├── words.txt             # input: word symbol table
│   ├── tokens.txt            # output: token symbol table
│   ├── lexicon.txt           # output: word→char mapping
│   ├── lexicon_disambig.txt  # output: with disambiguation symbols
│   ├── L.pt                  # output: lexicon FST
│   ├── L_disambig.pt         # output: lexicon FST with self-loops
│   └── HLG.pt                # output: final decoding graph
├── lang_bpe/                 # BPE workflow outputs
│   ├── words.txt             # output of prepare_words.py
│   ├── tokens.txt            # output of prepare_bpe.py (from AM units.txt + disambig)
│   ├── lexicon.txt           # output of prepare_bpe.py (word→BPE pieces)
│   ├── lexicon_disambig.txt  # output of prepare_bpe.py
│   ├── L.pt                  # output of prepare_bpe.py
│   ├── L_disambig.pt         # output of prepare_bpe.py
│   └── HLG.pt                # output of compile_hlg.py
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

### `prepare_words.py`

```
usage: prepare_words.py --text FILE [FILE ...] --lang-dir DIR [--min-count N]

arguments:
  --text        One or more plain-text files (one sentence per line, no utterance IDs)
  --lang-dir    Output directory; 'words.txt' is written here
  --min-count   Minimum word frequency to include (default: 1)
```

### `prepare_bpe.py`

```
usage: prepare_bpe.py --lang-dir DIR --bpe-model FILE --units-file FILE

arguments:
  --lang-dir    Input/output lang directory (must contain 'words.txt')
  --bpe-model   Path to the SentencePiece .model file (from the acoustic model)
  --units-file  Path to units.txt from the acoustic model (BPE token symbol table)
```

Requires: `sentencepiece` (`pip install sentencepiece`)

### `make_kn_lm.py`

```
usage: make_kn_lm.py [-ngram-order N] [-text FILE [FILE ...]] [-lm FILE]

arguments:
  -ngram-order  N-gram order, 1–7 (default: 3)
  -text         One or more input corpus files (plain text, one sentence per line,
                no utterance IDs); reads stdin if omitted
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

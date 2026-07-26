# Checkpoints — formats, conversion, and the native bundle

How OASR loads models: `oasr.from_pretrained(id_or_path)` accepts a local
checkpoint directory or a HuggingFace Hub repo id (downloaded via
`snapshot_download`, `oasr[hub]` extra) and resolves it to a registered model
architecture plus a **complete bundle** — weights, model config, tokenizer
spec, feature spec, and decoding defaults all travel together, so the engine
never sniffs paths or applies engine-side defaults that fight the checkpoint.

```python
import oasr
model, config = oasr.from_pretrained("/path/to/ckpt_dir")

# Engine path (what ASREngine uses) — also returns the specs + LoadReport:
from oasr.models import load_pretrained
loaded = load_pretrained("/path/to/ckpt_dir", device="cuda", dtype=torch.bfloat16)
loaded.model, loaded.config, loaded.tokenizer_spec, loaded.feature_spec
```

## Resolution precedence

1. **Native format** — the directory contains `oasr_config.json` → loaded
   directly (safetensors, no conversion, no third-party deps).
2. **Explicit override** — `from_pretrained(dir, architecture="transducer")`
   picks that converter, no sniffing.  Required for icefall
   pruned-transducer dirs (they sniff as `zipformer`, and hybrid checkpoints
   carry both branches).
3. **Converter detection** — every registered `CheckpointConverter.detect()`
   inspects the directory for its own format markers, and the claims are
   **ranked** by `detect_specificity`:

   | Level | Meaning | Declared by |
   |---|---|---|
   | `DETECT_KEYED_VALUE` (30) | a named config file whose field names the architecture (`config.json: model_type == "whisper"`, `config.yaml: model: Paraformer`) | `whisper`, `speech_llm`, `paraformer` |
   | `DETECT_NAMED_CONFIG` (20) | a framework-specific config file exists (WeNet's `train.yaml`) — identifies the framework, not the architecture | `conformer` |
   | `DETECT_ASSET_LAYOUT` (10) | filename / asset conventions only (`exp/` layout, `epoch-*.pt`, `tokens.txt` beside the weights) — the default for a converter that declares nothing | `zipformer`, `transducer` |

   The highest-specificity claim wins; a **tie** at the top raises, listing the
   candidates (pass `architecture=`).  **Zero** claims also raises now — the old
   `"conformer"` fallback guessed WeNet for anything unrecognized and then failed
   deep inside weight loading with a shape error, so the guess is refused where the
   information is actually missing.

   Ranking exists so each `detect()` can state **only positive markers**. Several
   formats share filenames — a FunASR dir carries a `model.pt`, a WeNet dir a
   `final.pt`, both of which satisfy icefall's asset rule — and the previous answer
   was `return False` guards *inside `IcefallConverter.detect`* naming WeNet's and
   FunASR's markers. That put one format's knowledge in another's converter, so a
   7th format meant editing an unrelated file. **When adding a converter, declare
   `detect_specificity` and never add a negative guard for another format.**

## Converter contract

A `CheckpointConverter` (registered together with its model via
`register_model(name, model_cls, config_cls, converter=...)`) implements
`detect` / `build_config` / `build_aux` / `load_state_dict`, and `convert()`
returns a `ConvertedCheckpoint` bundle:

| Field | Meaning |
|---|---|
| `architecture` | model registry key |
| `model_config` | `BaseModelConfig` subclass (shape/topology) |
| `aux` | side objects (e.g. CMVN stats) |
| `state_dict` | source-format weights (name mapping happens in `Model.load_weights`) |
| `tokenizer` | `TokenizerSpec` — kind + asset files + options (see `docs/tokenizers.md`) |
| `features` | `FeatureSpec` — feature kind, sample rate, mels, LFR, window, normalize, `audio_scale` |
| `decoding` | `DecodingDefaults` — default decode family, blank/sos/eos ids |

Legacy 4-method converters keep working through an adapter that fills the
metadata via the historical path-sniffing behaviour.

### Weight-load accounting

`Model.load_weights` returns a `LoadReport{mapped, dropped, missing}`.  The
registry warns about drops per the converter's declaration:
`expected_unused_prefixes` are silently fine (e.g. icefall's pruned-RNNT
`simple_am_proj` / `simple_lm_proj` training heads), while
`capability_drop_hints` produce a **named warning** describing the capability
being lost (e.g. dropping a U2++ `decoder.*` branch would lose
`ctc_aed_rescoring`).  Silent weight drops are a bug by policy.

## Built-in converters

| Ecosystem | Detect marker | Architecture | Notes |
|---|---|---|---|
| WeNet | `train.yaml` | `conformer` | U2/U2++ dirs with a `(bi)transformer` decoder keep the AED branch → `capabilities={"ctc","ctc_aed_rescoring"}` |
| icefall | `tokens.txt`/`bpe.model` + `epoch-*.pt`/`pretrained.pt` | `zipformer` (CTC) | config shape-inferred from the checkpoint |
| icefall pruned-transducer | **not auto-detected** — `architecture="transducer"` | `transducer` | `encoder_type ∈ {conformer, zipformer}`; config inferred from `decoder.*`/`joiner.*` |
| HF Whisper | `config.json: model_type=whisper` | `whisper` | emits `whisper` tokenizer + `whisper_logmel` features (`audio_scale=1.0`) |
| HF Qwen2-Audio | `config.json: model_type=qwen2_audio` | `speech_llm` | fills omitted `text_config` fields from Qwen2 defaults; sharded safetensors |
| FunASR Paraformer | `config.yaml: model: Paraformer` | `paraformer` | parses `am.mvn` CMVN into synthetic state-dict buffers; `funasr_char` tokenizer; LFR 7/6 features |

## Native format

`oasr-convert <src> <dst>` (also `python -m oasr.checkpoints.convert`)
materializes any supported directory as a round-trippable native bundle:

```
dst/
├── oasr_config.json      # format_version 1: architecture + model config +
│                         # tokenizer/feature/decoding specs
├── model.safetensors     # post-load_weights state dict — loads via strict
│                         # load_state_dict, no name mapping
└── tokenizer/            # tokenizer assets copied verbatim
```

Properties worth knowing:

- Loads need **no third-party ecosystem installed** (no wenet / icefall /
  transformers / funasr on the serving host) and mmap via safetensors.
- Model-declared computed buffers (e.g. Conformer's `pos_enc.pe`) are listed
  in `_computed_buffer_suffixes` and skipped at save; they rebuild at load.
- The model's own state-dict `_metadata` is stamped at load so version-gated
  `_load_from_state_dict` remaps don't re-fire on native round-trips.
- Bundles load host-side (`map_location="cpu"`) and the engine casts dtype
  **before** the device move — a GPU-mapped bundle would hold a full second
  weight copy resident while the model moves over (an 8.4 B-param checkpoint
  cannot load on a 32 GB card otherwise).

# Tokenizers — the sixth registry axis

Text production (and, for AR families, prompt **encoding**) is a pluggable
axis: a `Tokenizer` implementation is selected by the checkpoint converter's
`TokenizerSpec`, never by engine-side path sniffing.

```python
class Tokenizer(ABC):
    vocab_size: int
    special_ids: FrozenSet[int]          # stripped from decode output

    def decode(self, ids: Sequence[int]) -> str: ...
    def encode(self, text: str) -> List[int]: ...   # AED prompts / LLM / hotwords
```

A `TokenizerSpec` is `kind` + asset `files` + `options`; `build_tokenizer(spec)`
resolves the kind through the registry (`register_tokenizer(kind, factory)`).

## Built-in kinds

| Kind | Assets | Used by | Behaviour |
|---|---|---|---|
| `symbol_table` | `units.txt` / `tokens.txt` | WeNet CTC, icefall CTC | id→piece join, `▁` = word boundary; default `special_ids={0,1,2}` — **bit-compatible** with the legacy `Detokenizer` |
| `sentencepiece` | `bpe.model` | icefall BPE | ids == SentencePiece piece ids (`oasr[tokenizers]` extra) |
| `huggingface` | `tokenizer.json` (+ `tokenizer_config.json`) | Qwen2-Audio (speech-LLM) | wraps the `tokenizers` fast runtime; see the added-token merge below |
| `whisper` | `tokenizer.json` | HF Whisper | HF fast tokenizer + control-token stripping above `eot_id` |
| `funasr_char` | `tokens.json` | Paraformer | ports FunASR's `sentence_postprocess` (CJK join, `@@` subword merge, abbreviation join) |

## The `tokenizer_config.json` added-token merge (Qwen2-Audio trap)

Some HF checkpoints declare added special tokens **only** in
`tokenizer_config.json: added_tokens_decoder` — Qwen2-Audio's audio and
timestamp specials (3288 entries, ids 151643+) are absent from
`tokenizer.json`.  `transformers` merges them at load; the raw `tokenizers`
runtime does not, and prompt encoding silently breaks (`<|audio_bos|>`
BPE-splits into plain text).  When the spec carries a `tokenizer_config`
file, `HuggingFaceTokenizer` adds the missing entries the same way:
in declared-id order (relying on the backend's sequential id assignment),
with a loud warning if any id lands elsewhere.

## Relationship to the engine

`oasr/engine/decode/detokenize.py::Detokenizer` is a thin adapter over this
axis kept for backward compatibility: the engine injects the spec-built
tokenizer when the checkpoint provides one; the legacy sniffed
`unit_table` / `sentencepiece_model` paths build the same `symbol_table`
tokenizer and remain decode-for-decode identical to the historical
behaviour.  Decode strategies reach the tokenizer via `detok.tokenizer`
(the `llm` strategy requires `encode` for its chat template).

Per-request n-best transcripts (`DecodingOptions.n_best` /
`RequestOutput.nbest_texts`) are detokenized through the same tokenizer by
the executors on final outputs.

## Which direction each kind supports

`decode` is universal; `encode` is not.  A `symbol_table` tokenizer holds only an
id→piece map, so it can render output but cannot turn a prompt back into ids —
which is why `Tokenizer.supports_encode` exists.  Test **that**, never
`hasattr(tok, "encode")`: `encode` is abstract on the ABC, so the attribute is
always present even on the kinds that raise from it.

| kind | `supports_encode` | `special_ids` reports |
|---|---|---|
| `symbol_table` | ✗ (decode-only) | `{0, 1, 2}` |
| `sentencepiece` | ✓ | `{0, 1, 2}` |
| `huggingface` | ✓ | spec-provided (stripping delegates to `skip_special_tokens`) |
| `whisper` | ✓ | the whole control block, `eot_id … vocab_size` |
| `funasr_char` | ✓ | spec-provided |

`special_ids` means exactly *what `decode` strips*, so filtering a hypothesis by it
yields the tokens `decode` would keep.  Whisper's `decode` drops every id at or
above `eot_id` (language / task / timestamp markers), so it reports that whole
range — reporting only `{eot_id}` would let a caller leak control markup.

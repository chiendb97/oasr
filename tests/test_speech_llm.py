# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Speech-LLM (Qwen2-Audio) tests: converter, tokenizer, parity, strategy, engine.

Checkpoint-dependent tests use the tiny random fixture at ``SPEECH_LLM_TINY``
(real Qwen2-Audio tokenizer + 2+2-layer random weights, built by the Phase 4
fixture script; HF reference outputs under ``oasr_ref/ref.pt``) and skip when
absent.  ``TestRealCheckpoint`` additionally needs the full
Qwen2-Audio-7B-Instruct snapshot at ``SPEECH_LLM_CKPT``.
"""

import os

import assets
import pytest
import torch

# Declared once in tests/assets.py; the markers below gate through the same
# path, so --strict-assets can turn "fixture absent" into a failure.
SPEECH_LLM_TINY = assets.declared("SPEECH_LLM_TINY")
SPEECH_LLM_CKPT = assets.declared("SPEECH_LLM_CKPT")
WAV_DIR = assets.declared("WAV_DIR")

needs_tiny = pytest.mark.requires_assets("SPEECH_LLM_TINY")
needs_tiny_ref = pytest.mark.requires_assets("SPEECH_LLM_TINY", "SPEECH_LLM_TINY_REF")
needs_real = pytest.mark.requires_assets("SPEECH_LLM_CKPT")


# ---------------------------------------------------------------------------
# Config / registry seams (no checkpoint)
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults_match_qwen2_audio_7b(self):
        from oasr.models.speech_llm import SpeechLlmModelConfig

        cfg = SpeechLlmModelConfig()
        assert cfg.model_type == "speech_llm"
        assert (cfg.audio_d_model, cfg.audio_encoder_layers, cfg.audio_num_mel_bins) == (
            1280,
            32,
            128,
        )
        assert cfg.audio_head_dim == 64
        assert cfg.eos_token_ids == [151643, 151645]
        assert "{prompt}" in cfg.prompt_suffix

    def test_from_dict_round_trip(self):
        import dataclasses

        from oasr.models.speech_llm import SpeechLlmModelConfig

        cfg = SpeechLlmModelConfig(vocab_size=1234, text_num_key_value_heads=4)
        d = dataclasses.asdict(cfg)
        cfg2 = SpeechLlmModelConfig.from_dict(d)
        assert cfg2 == cfg

    def test_model_metadata(self):
        from oasr.models.speech_llm import SpeechLlmModel, SpeechLlmModelConfig

        cfg = SpeechLlmModelConfig(
            vocab_size=100,
            audio_num_mel_bins=8,
            audio_d_model=16,
            audio_encoder_layers=1,
            audio_encoder_attention_heads=2,
            audio_encoder_ffn_dim=32,
            audio_max_source_positions=10,
            text_hidden_size=32,
            text_num_hidden_layers=1,
            text_num_attention_heads=4,
            text_num_key_value_heads=2,
            text_intermediate_size=64,
        )
        m = SpeechLlmModel(cfg)
        assert m.default_decode_type == "llm"
        assert m.capabilities == frozenset({"llm"})
        assert m.streaming_kind == "none"
        assert m.decoder is m.language_model

    def test_canonical_key_layouts(self):
        from oasr.models.speech_llm.model import SpeechLlmModel

        f = SpeechLlmModel._canonical_key
        # HF 4.x published layout
        assert f("language_model.model.layers.0.mlp.up_proj.weight") == (
            "language_model.layers.0.mlp.up_proj.weight"
        )
        assert f("language_model.lm_head.weight") == "language_model.lm_head.weight"
        # HF 5.x resave (double-nested trunk)
        assert f("language_model.model.model.norm.weight") == "language_model.norm.weight"
        # 5.x internal layout with top-level lm_head
        assert f("model.audio_tower.conv1.weight") == "audio_tower.conv1.weight"
        assert f("lm_head.weight") == "language_model.lm_head.weight"
        # native (canonical) passes through
        assert f("audio_tower.layers.3.fc1.bias") == "audio_tower.layers.3.fc1.bias"


class TestTowerLengths:
    def test_hf_two_stage_formula(self):
        from oasr.models.speech_llm.audio_tower import Qwen2AudioTower

        mel = torch.tensor([3000, 300, 190, 1])
        feat = Qwen2AudioTower.feat_lengths(mel)
        out = Qwen2AudioTower.output_lengths(mel)
        assert feat.tolist() == [1500, 150, 95, 1]
        assert out.tolist() == [750, 75, 47, 0]

    def test_whisper_logmel_true_lengths(self):
        from oasr.features import FeatureConfig
        from oasr.features.whisper import batched_whisper_logmel

        cfg = FeatureConfig(feature_type="whisper_logmel", num_mel_bins=128, dither=0.0)
        wav = torch.randn(2, 16000 * 3)
        lengths = torch.tensor([16000 * 3, 16000 + 1])
        feats, flens = batched_whisper_logmel(wav, lengths, cfg)
        assert feats.shape == (2, 3000, 128)
        # ceil(len / hop): 48000/160 = 300; (16001 + 159) // 160 = 101.
        assert flens.tolist() == [300, 101]


# ---------------------------------------------------------------------------
# Converter / bundle / native format
# ---------------------------------------------------------------------------


@needs_tiny
class TestConverter:
    def test_detect_and_precedence(self):
        from oasr.models.registry import resolve_architecture
        from oasr.models.speech_llm import HFQwen2AudioConverter

        conv = HFQwen2AudioConverter()
        assert conv.detect(SPEECH_LLM_TINY)
        assert resolve_architecture(SPEECH_LLM_TINY) == "speech_llm"

    def test_detect_rejects_whisper(self, tmp_path):
        import json

        from oasr.models.speech_llm import HFQwen2AudioConverter

        (tmp_path / "config.json").write_text(json.dumps({"model_type": "whisper"}))
        assert not HFQwen2AudioConverter().detect(tmp_path)

    def test_bundle_complete(self):
        from oasr.models.registry import load_checkpoint_bundle

        arch, bundle = load_checkpoint_bundle(SPEECH_LLM_TINY)
        assert arch == "speech_llm"
        assert bundle.source_format == "huggingface"
        assert bundle.tokenizer.kind == "huggingface"
        assert "tokenizer_config" in bundle.tokenizer.files  # added-token source
        f = bundle.features
        assert (f.kind, f.feature_dim, f.audio_scale) == ("whisper_logmel", 128, 1.0)
        assert bundle.decoding.default_decode_type == "llm"
        cfg = bundle.model_config
        # tiny fixture dims + generation ids
        assert (cfg.text_hidden_size, cfg.text_num_key_value_heads) == (64, 2)
        assert cfg.audio_token_id == 151646
        assert cfg.eos_token_ids == [151643, 151645]

    def test_load_report_clean(self):
        from oasr.models.registry import instantiate_from_bundle, load_checkpoint_bundle

        arch, bundle = load_checkpoint_bundle(SPEECH_LLM_TINY)
        model, cfg, report = instantiate_from_bundle(arch, bundle)
        assert not report.missing
        assert not report.dropped
        assert sorted(model.capabilities) == ["llm"]

    def test_native_round_trip(self, tmp_path):
        pytest.importorskip("safetensors")
        from oasr.checkpoints.convert import convert_to_native
        from oasr.models.registry import instantiate_from_bundle, load_checkpoint_bundle

        out = tmp_path / "native"
        convert_to_native(SPEECH_LLM_TINY, str(out))
        arch, bundle = load_checkpoint_bundle(out)
        assert (arch, bundle.source_format) == ("speech_llm", "native")
        assert bundle.tokenizer.kind == "huggingface"
        m2, cfg2, _ = instantiate_from_bundle(arch, bundle)

        arch1, b1 = load_checkpoint_bundle(SPEECH_LLM_TINY)
        m1, _, _ = instantiate_from_bundle(arch1, b1)
        sd1, sd2 = m1.state_dict(), m2.state_dict()
        assert set(sd1) == set(sd2)
        for k in sd1:
            assert torch.equal(sd1[k], sd2[k]), k


# ---------------------------------------------------------------------------
# Tokenizer: tokenizer_config added-token merge
# ---------------------------------------------------------------------------


@needs_tiny
class TestTokenizerAddedTokens:
    def _tok(self):
        from oasr.models.registry import load_checkpoint_bundle
        from oasr.tokenizers import build_tokenizer

        _, bundle = load_checkpoint_bundle(SPEECH_LLM_TINY)
        return build_tokenizer(bundle.tokenizer)

    def test_audio_specials_encode_as_single_tokens(self):
        tok = self._tok()
        assert tok.encode("<|AUDIO|>") == [151646]
        assert tok.encode("<|audio_bos|>") == [151647]
        assert tok.encode("<|audio_eos|>") == [151648]

    def test_specials_stripped_from_decode(self):
        tok = self._tok()
        ids = tok.encode("hello world")
        assert tok.decode(ids + [151645, 151643]) == "hello world"


# ---------------------------------------------------------------------------
# Parity vs the HF reference (tiny fixture, CPU fp32)
# ---------------------------------------------------------------------------


@needs_tiny
@needs_tiny_ref
class TestHFParityTinyFixture:
    @pytest.fixture(scope="class")
    def ref(self):
        return torch.load(os.path.join(SPEECH_LLM_TINY, "oasr_ref", "ref.pt"), weights_only=False)

    @pytest.fixture(scope="class")
    def loaded(self):
        from oasr.models.loaders import load_pretrained

        return load_pretrained(SPEECH_LLM_TINY, device="cpu", dtype=torch.float32)

    @pytest.fixture(scope="class")
    def prompt_ids(self, loaded):
        from oasr.models.speech_llm.config import (
            QWEN2_AUDIO_PROMPT_PREFIX,
            QWEN2_AUDIO_PROMPT_SUFFIX,
        )
        from oasr.tokenizers import build_tokenizer

        tok = build_tokenizer(loaded.tokenizer_spec)
        prefix = tok.encode(QWEN2_AUDIO_PROMPT_PREFIX)
        suffix = tok.encode(
            QWEN2_AUDIO_PROMPT_SUFFIX.format(prompt=loaded.config.default_user_prompt)
        )
        return prefix, suffix

    def test_features_match_hf(self, ref, loaded):
        import torchaudio

        from oasr.features.whisper import batched_whisper_logmel

        fcfg = loaded.feature_spec.to_feature_config()
        for i, row in enumerate(ref["rows"]):
            a = torchaudio.load(ref["wavs"][i])[0].squeeze(0)
            if i == 0:
                a = a[: 16000 * 3]
            feats, flens = batched_whisper_logmel(a.unsqueeze(0), torch.tensor([a.numel()]), fcfg)
            hf_mel = row["input_features"].transpose(1, 2)
            assert (feats - hf_mel).abs().max().item() < 5e-4
            assert int(flens[0]) == int(row["feature_attention_mask"].sum())

    def test_tower_projector_match_hf(self, ref, loaded):
        for row in ref["rows"]:
            mel = row["input_features"].transpose(1, 2)
            mlen = torch.tensor([int(row["feature_attention_mask"].sum())])
            with torch.no_grad():
                emb, elens = loaded.model.encode_offline(mel, mlen)
            L = int(elens[0])
            assert (emb[0, :L] - row["proj_out"][0, :L]).abs().max().item() < 1e-5

    def test_prompt_ids_match_processor(self, ref, loaded, prompt_ids):
        prefix, suffix = prompt_ids
        assert ref["prompt"] == loaded.config.default_user_prompt
        for row in ref["rows"]:
            hf_ids = row["input_ids"][0].tolist()
            n_audio = sum(1 for t in hf_ids if t == loaded.config.audio_token_id)
            ours = prefix + [loaded.config.audio_token_id] * n_audio + suffix
            assert ours == hf_ids

    def _greedy(self, loaded, prompt_ids, mels, mlens, max_new=24):
        prefix_ids, suffix_ids = prompt_ids
        model = loaded.model
        eos = set(loaded.config.eos_token_ids)
        with torch.no_grad():
            emb, elens = model.encode_offline(torch.cat(mels), torch.tensor(mlens))
        dec = model.decoder
        prefix = dec.embed_tokens(torch.tensor(prefix_ids))
        suffix = dec.embed_tokens(torch.tensor(suffix_ids))
        B = len(mels)
        totals = [len(prefix_ids) + int(elens[i]) + len(suffix_ids) for i in range(B)]
        P = max(totals)
        inputs = torch.zeros(B, P, emb.size(2))
        valid = torch.zeros(B, P, dtype=torch.bool)
        for i in range(B):
            inputs[i, P - totals[i] :] = torch.cat([prefix, emb[i, : int(elens[i])], suffix], dim=0)
            valid[i, P - totals[i] :] = True
        tokens = [[] for _ in range(B)]
        alive = list(range(B))
        with torch.no_grad():
            logits, state = dec.prefill(inputs, valid)
            for _ in range(max_new):
                nxt = logits.float().argmax(-1)
                done = []
                for r, t in enumerate(nxt.tolist()):
                    tokens[alive[r]].append(int(t))
                    if t in eos:
                        done.append(r)
                keep = [r for r in range(len(alive)) if r not in done]
                alive = [alive[r] for r in keep]
                if not alive:
                    break
                if done:
                    ki = torch.tensor(keep)
                    state = dec.select(state, ki)
                    nxt = nxt.index_select(0, ki)
                logits, state = dec.step(nxt, state)
        return tokens

    def test_greedy_token_exact_single(self, ref, loaded, prompt_ids):
        for row in ref["rows"]:
            mel = row["input_features"].transpose(1, 2)
            mlen = int(row["feature_attention_mask"].sum())
            ours = self._greedy(loaded, prompt_ids, [mel], [mlen])[0]
            assert ours == row["generated"]

    def test_greedy_token_exact_batched_left_padded(self, ref, loaded, prompt_ids):
        mels = [r["input_features"].transpose(1, 2) for r in ref["rows"]]
        mlens = [int(r["feature_attention_mask"].sum()) for r in ref["rows"]]
        ours = self._greedy(loaded, prompt_ids, mels, mlens)
        for i in range(len(ref["rows"])):
            assert ours[i] == ref["batched"]["generated"][i]


# ---------------------------------------------------------------------------
# Strategy over the incremental protocol (CPU, tiny fixture)
# ---------------------------------------------------------------------------


@needs_tiny
class TestLlmStrategy:
    def _strategy_and_model(self, **cfg_kwargs):
        from oasr.engine.config import EngineConfig
        from oasr.engine.decode import Detokenizer, build_decode_strategy
        from oasr.models.loaders import load_pretrained
        from oasr.tokenizers import build_tokenizer

        loaded = load_pretrained(SPEECH_LLM_TINY, device="cpu", dtype=torch.float32)
        cfg = EngineConfig(ckpt_dir=SPEECH_LLM_TINY, service_mode="offline", **cfg_kwargs)
        detok = Detokenizer(tokenizer=build_tokenizer(loaded.tokenizer_spec))
        strategy = build_decode_strategy("llm", cfg, detok, loaded.model)
        return strategy, loaded.model

    def test_kv_storage_auto_pages_only_where_steps_are_captured(self):
        """``auto`` is a trade, not a preference.

        Paging costs the block-table indirection and buys no capacity while
        admission reserves each row's ceiling; what it buys is a step CUDA graph,
        which needs addresses that outlive a batch.  So it must follow
        ``step_graphs`` — on for a decoder that declares it, off otherwise —
        rather than being chosen because paging sounds better.  An explicit
        value still wins, which is what makes the A/B possible.
        """
        strategy, _ = self._strategy_and_model()
        assert strategy.options.kv_storage == "auto"
        assert strategy._decoder().supports_step_graphs

        # Three conditions, and all three have to hold.  This fixture loads on
        # CPU, where there is no graph to capture and paging would be pure
        # indirection on a path production never runs — so the device is one of
        # them, and the CUDA case is checked below on the same weights.
        off, _ = self._strategy_and_model(decode_options={"step_graphs": False})
        assert off._resolve_storage(off._decoder()) == "dense"
        on, _ = self._strategy_and_model(decode_options={"step_graphs": True})
        assert on._resolve_storage(on._decoder()) == "dense"  # CPU

        if torch.cuda.is_available():
            cuda_on, model = self._strategy_and_model(decode_options={"step_graphs": True})
            model.cuda()
            assert cuda_on._resolve_storage(cuda_on._decoder()) == "paged"
            cuda_off, model = self._strategy_and_model(decode_options={"step_graphs": False})
            model.cuda()
            assert cuda_off._resolve_storage(cuda_off._decoder()) == "dense"

        # An explicit value wins in both directions, which is what makes the A/B
        # possible at all.
        forced, _ = self._strategy_and_model(
            decode_options={"step_graphs": True, "kv_storage": "dense"}
        )
        assert forced._resolve_storage(forced._decoder()) == "dense"
        assert forced.kv_manager() is None
        asked, _ = self._strategy_and_model(decode_options={"kv_storage": "paged"})
        assert asked._resolve_storage(asked._decoder()) == "paged"

    def test_beam_search_stays_dense_even_asking_for_graphs(self):
        """A paged row's pages are its own, and a beam grid reorders onto its
        parents with repeated ``select`` indices — which would alias two slots
        onto one page.  Storage is decided for the strategy, so the refusal
        happens once here rather than being discovered mid-search."""
        strat, _ = self._strategy_and_model(
            decode_options={"step_graphs": True, "kv_storage": "paged", "beam_size": 2}
        )
        assert strat.kv_manager() is None

    def test_partials_then_finals(self):
        from types import SimpleNamespace

        from oasr.engine.generation import StepBudget

        strategy, model = self._strategy_and_model(max_new_tokens=12)
        mel = torch.randn(2, 3000, 128)
        mlens = torch.tensor([300, 190])
        with torch.no_grad():
            emb, elens = model.encode_offline(mel, mlens)
        reqs = [SimpleNamespace(request_id=f"r{i}") for i in range(2)]
        strategy.begin_offline(reqs, emb, elens)
        assert strategy.has_pending()

        partials, finals = {}, {}
        for _ in range(30):
            for out in strategy.advance(StepBudget(max_steps=4)):
                if out.finished:
                    finals[out.request_id] = out
                else:
                    partials.setdefault(out.request_id, []).append(len(out.tokens[0]))
            if not strategy.has_pending():
                break
        assert not strategy.has_pending()
        assert set(finals) == {"r0", "r1"}
        for rid in ("r0", "r1"):
            assert partials[rid], "no partials emitted"
            assert partials[rid] == sorted(partials[rid]), "non-monotonic partials"
            assert len(finals[rid].tokens[0]) <= 12

    def test_late_arrivals_are_absorbed_into_one_group(self):
        """A second prefill joins the group already generating.

        The transcripts must be *identical* to running with ``merge_groups=0``:
        merging changes how many decoder forwards it takes, never what is
        decoded.  Which is also the point — two groups cost about twice one group
        of the same total rows, because an AR step is weight-read bound.
        """
        from types import SimpleNamespace

        from oasr.engine.generation import StepBudget

        mel = torch.randn(3, 3000, 128)
        mlens = torch.tensor([300, 190, 240])

        def run(**opts):
            strategy, model = self._strategy_and_model(max_new_tokens=10, **opts)
            with torch.no_grad():
                emb, elens = model.encode_offline(mel, mlens)
            reqs = [SimpleNamespace(request_id=f"r{i}") for i in range(3)]
            strategy.begin_offline(reqs[:2], emb[:2], elens[:2])
            strategy.advance(StepBudget(max_steps=3))  # get the first group moving
            strategy.begin_offline(reqs[2:], emb[2:], elens[2:])
            groups = len(strategy._groups)
            finals = {}
            for _ in range(40):
                for out in strategy.advance(StepBudget(max_steps=4)):
                    if out.finished:
                        finals[out.request_id] = out.tokens[0]
                if not strategy.has_pending():
                    break
            assert not strategy.has_pending()
            return groups, finals

        merged_groups, merged = run()
        split_groups, split = run(decode_options={"merge_groups": False})
        assert merged_groups == 1 and split_groups == 2
        assert set(merged) == set(split) == {"r0", "r1", "r2"}
        for rid in merged:
            assert merged[rid] == split[rid], f"{rid} decoded differently after a merge"

    def test_free_session_aborts_row(self):
        from types import SimpleNamespace

        from oasr.engine.generation import StepBudget

        strategy, model = self._strategy_and_model(max_new_tokens=8)
        mel = torch.randn(2, 3000, 128)
        with torch.no_grad():
            emb, elens = model.encode_offline(mel, torch.tensor([300, 190]))
        reqs = [SimpleNamespace(request_id=f"r{i}") for i in range(2)]
        strategy.begin_offline(reqs, emb, elens)
        strategy.free_session(reqs[0])
        finals = set()
        for _ in range(20):
            for out in strategy.advance(StepBudget(max_steps=8)):
                if out.finished:
                    finals.add(out.request_id)
            if not strategy.has_pending():
                break
        assert finals == {"r1"}

    def test_llm_prompt_override(self):
        strategy_default, _ = self._strategy_and_model()
        strategy_custom, _ = self._strategy_and_model(llm_prompt="Transcribe.")
        assert strategy_custom._default_suffix_ids != strategy_default._default_suffix_ids
        assert strategy_custom._prefix_ids == strategy_default._prefix_ids

    def test_per_request_prompt_override(self):
        from types import SimpleNamespace

        from oasr.engine.request import DecodingOptions

        strategy, _ = self._strategy_and_model()
        default_req = SimpleNamespace(request_id="r0", decoding=None)
        custom_req = SimpleNamespace(
            request_id="r1", decoding=DecodingOptions(prompt="Transcribe.")
        )
        default_ids = strategy._suffix_ids_for(default_req)
        custom_ids = strategy._suffix_ids_for(custom_req)
        assert default_ids == strategy._default_suffix_ids
        assert custom_ids != default_ids
        # Memoised: same list object on repeat lookup.
        assert strategy._suffix_ids_for(custom_req) is custom_ids

    def test_per_request_max_new_tokens_and_finish_reason(self):
        from types import SimpleNamespace

        from oasr.engine.generation import StepBudget
        from oasr.engine.request import DecodingOptions

        # Engine default 64, request caps row 1 at 3 — the capped row must
        # stop with finish_reason="length" after exactly 3 tokens (the tiny
        # random fixture never emits EOS that early).
        strategy, model = self._strategy_and_model(max_new_tokens=64)
        mel = torch.randn(2, 3000, 128)
        with torch.no_grad():
            emb, elens = model.encode_offline(mel, torch.tensor([300, 190]))
        reqs = [
            SimpleNamespace(request_id="r0", decoding=None),
            SimpleNamespace(request_id="r1", decoding=DecodingOptions(max_new_tokens=3)),
        ]
        strategy.begin_offline(reqs, emb, elens)
        finals = {}
        for _ in range(100):
            for out in strategy.advance(StepBudget(max_steps=8)):
                if out.finished:
                    finals[out.request_id] = out
            if "r1" in finals:
                break
        assert "r1" in finals
        assert len(finals["r1"].tokens[0]) == 3
        assert finals["r1"].finish_reason == "length"

    def test_preallocated_kv_matches_cat_growth(self):
        """The in-place preallocated KV path (``prefill(capacity=...)``) must
        be logit- and token-identical to the legacy cat-growth path, across
        step and select (row drop)."""
        from oasr.models.loaders import load_pretrained

        loaded = load_pretrained(SPEECH_LLM_TINY, device="cpu", dtype=torch.float32)
        lm = loaded.model.language_model
        torch.manual_seed(0)
        B, P, D = 3, 17, loaded.model.config.text_hidden_size
        emb = torch.randn(B, P, D)
        valid = torch.ones(B, P, dtype=torch.bool)
        valid[0, :5] = False
        valid[2, :2] = False

        def run(capacity):
            with torch.no_grad():
                logits, state = lm.prefill(emb, valid, capacity=capacity)
                seq = [logits.clone()]
                toks = logits.argmax(-1)
                for step in range(6):
                    logits, state = lm.step(toks, state)
                    seq.append(logits.clone())
                    toks = logits.argmax(-1)
                    if step == 2:  # drop the middle row mid-generation
                        keep = torch.tensor([0, 2])
                        state = lm.select(state, keep)
                        toks = toks.index_select(0, keep)
            return seq

        legacy = run(None)
        prealloc = run(P + 16)
        for a, b in zip(legacy, prealloc):
            torch.testing.assert_close(a, b)

    def test_preallocated_kv_overflow_falls_back(self):
        """Stepping past the declared capacity must degrade to cat-growth,
        not corrupt the cache."""
        from oasr.models.loaders import load_pretrained

        loaded = load_pretrained(SPEECH_LLM_TINY, device="cpu", dtype=torch.float32)
        lm = loaded.model.language_model
        torch.manual_seed(0)
        P, D = 9, loaded.model.config.text_hidden_size
        emb = torch.randn(1, P, D)
        valid = torch.ones(1, P, dtype=torch.bool)

        def run(capacity):
            with torch.no_grad():
                logits, state = lm.prefill(emb, valid, capacity=capacity)
                out = [logits.clone()]
                toks = logits.argmax(-1)
                for _ in range(5):
                    logits, state = lm.step(toks, state)
                    out.append(logits.clone())
                    toks = logits.argmax(-1)
            return out

        legacy = run(None)
        tight = run(P + 2)  # overflows on the 3rd step
        for a, b in zip(legacy, tight):
            torch.testing.assert_close(a, b)

    def test_sampling_topk1_matches_greedy(self):
        from types import SimpleNamespace

        from oasr.engine.generation import StepBudget
        from oasr.engine.request import DecodingOptions

        # temperature>0 with top_k=1 is argmax-by-construction, so the sampled
        # row must produce exactly the greedy transcript — this exercises the
        # sampling code path deterministically.
        def run(decoding):
            strategy, model = self._strategy_and_model(max_new_tokens=8)
            torch.manual_seed(0)
            mel = torch.zeros(1, 3000, 128)
            with torch.no_grad():
                emb, elens = model.encode_offline(mel, torch.tensor([300]))
            req = SimpleNamespace(request_id="r0", decoding=decoding)
            strategy.begin_offline([req], emb, elens)
            for _ in range(50):
                for out in strategy.advance(StepBudget(max_steps=8)):
                    if out.finished:
                        return out.tokens[0]
            raise AssertionError("did not finish")

        greedy = run(None)
        sampled = run(DecodingOptions(temperature=2.0, top_k=1))
        assert sampled == greedy


# ---------------------------------------------------------------------------
# Engine end-to-end (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="engine requires CUDA")
@needs_tiny
class TestEngineE2E:
    def _audios(self, n=2):
        import torchaudio

        wavs = assets.require_wavs(n)
        return [torchaudio.load(w)[0].squeeze(0) for w in wavs]

    def test_offline_with_streaming_partials(self):
        from oasr.engine import ASREngine, EngineConfig

        audios = self._audios()
        cfg = EngineConfig(
            ckpt_dir=SPEECH_LLM_TINY,
            service_mode="offline",
            max_new_tokens=24,
            decode_steps_per_tick=4,
        )
        eng = ASREngine(cfg)
        ids = [eng.add_request(a, sample_rate=16000, streaming=False) for a in audios]
        partials, finals = {}, {}
        for _ in range(400):
            for o in eng.step():
                if o.finished:
                    finals[o.request_id] = o
                else:
                    partials.setdefault(o.request_id, []).append(len(o.tokens[0]))
            if len(finals) == len(ids):
                break
        assert len(finals) == len(ids)
        for rid in ids:
            assert partials.get(rid), "no token-streaming partials"
            assert partials[rid] == sorted(partials[rid])
            assert len(finals[rid].tokens[0]) <= 24

    def test_streaming_service_mode_rejected(self):
        from oasr.engine import ASREngine, EngineConfig

        with pytest.raises(ValueError, match="offline-only"):
            ASREngine(EngineConfig(ckpt_dir=SPEECH_LLM_TINY, service_mode="streaming"))

    def test_foreign_decode_method_rejected(self):
        from oasr.engine import ASREngine, EngineConfig

        with pytest.raises(ValueError, match="not a capability"):
            ASREngine(
                EngineConfig(ckpt_dir=SPEECH_LLM_TINY, service_mode="offline", decode_method="ctc")
            )

    def test_sustained_backlog_bounded_ticks_no_starvation(self):
        """Dispatcher-contract test under a sustained AR backlog.

        A one-model-per-process deployment's "mixed load" is a stream of LLM
        offline requests whose generations overlap fresh admissions.  The
        serving dispatcher runs ``engine.step()`` synchronously per tick, so
        the engine must guarantee (1) bounded work per tick — no request's
        transcript may grow by more than ``decode_steps_per_tick`` tokens
        between consecutive ticks; (2) the in-flight decode pool stays capped
        by ``max_decode_slots`` (+ one just-admitted batch); (3) late
        arrivals are neither dead-locked nor starved — everything finishes.
        """
        from oasr.engine import ASREngine, EngineConfig

        audios = self._audios()
        cfg = EngineConfig(
            ckpt_dir=SPEECH_LLM_TINY,
            service_mode="offline",
            max_batch_size=2,
            max_new_tokens=16,
            decode_steps_per_tick=2,
            max_decode_slots=2,
        )
        eng = ASREngine(cfg)
        n_req = 6
        ids = [
            eng.add_request(audios[i % len(audios)], sample_rate=16000, streaming=False)
            for i in range(n_req)
        ]
        finals = {}
        last_len = dict.fromkeys(ids, 0)
        for _ in range(600):
            outs = eng.step()
            # (2) pending decode pool bounded: slots + at most one fresh batch.
            assert eng.num_running <= cfg.max_decode_slots + cfg.max_batch_size
            for o in outs:
                # (1) bounded per-tick progress per request.
                grown = len(o.tokens[0]) - last_len[o.request_id]
                assert grown <= cfg.decode_steps_per_tick, (
                    f"request advanced {grown} tokens in one tick "
                    f"(budget {cfg.decode_steps_per_tick})"
                )
                last_len[o.request_id] = len(o.tokens[0])
                if o.finished:
                    finals[o.request_id] = o
            if len(finals) == n_req:
                break
        # (3) no starvation / deadlock.
        assert set(finals) == set(ids)


# ---------------------------------------------------------------------------
# Real checkpoint (GPU, 16 GB VRAM) — transcript sanity vs LJSpeech truth
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="engine requires CUDA")
@needs_real
class TestRealCheckpoint:
    def test_engine_transcribes_lj(self):
        import torchaudio

        from oasr.engine import ASREngine, EngineConfig

        wavs = assets.require_wavs(2)
        audios = [torchaudio.load(w)[0].squeeze(0) for w in wavs]
        cfg = EngineConfig(ckpt_dir=SPEECH_LLM_CKPT, service_mode="offline", max_new_tokens=96)
        eng = ASREngine(cfg)
        texts = eng.transcribe_offline(audios)
        # LJ001-0001/0002 ground truth inside the model's answer sentence
        # (exact HF-vs-OASR token match is not asserted in bf16 — greedy
        # near-ties on filler tokens flip; the recognized content must not).
        assert (
            "printing in the only sense with which we are at present concerned" in texts[0].lower()
        )
        assert "in being comparatively modern" in texts[1].lower()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestLmDecoderMerge:
    """Two speech-LLM groups joined into one forward.

    Harder than the AED case in one way: prompts are variable-length and
    *left*-padded, so the two groups differ in both the generation offset and the
    left-pad width, and the merged batch has to keep each row's own window.
    """

    @staticmethod
    def _lm():
        from oasr.models.speech_llm.config import SpeechLlmModelConfig
        from oasr.models.speech_llm.llm import Qwen2Lm

        torch.manual_seed(0)
        cfg = SpeechLlmModelConfig(
            vocab_size=64,
            text_hidden_size=2 * 16,
            text_num_attention_heads=2,
            text_num_key_value_heads=2,
            text_num_hidden_layers=2,
            text_intermediate_size=48,
        )
        return Qwen2Lm(cfg).eval()

    @staticmethod
    def _prompt(lm, batch, width, pad):
        """``(embeds, valid)`` for a left-padded prompt of ``width - pad`` tokens."""
        ids = torch.arange(batch * width).remainder(64).view(batch, width)
        valid = torch.ones(batch, width, dtype=torch.bool)
        valid[:, :pad] = False
        return lm.embed_tokens(ids), valid

    def _run(self, lm, prompt, steps, feed, start=0):
        logits, state = lm.prefill(*prompt, capacity=48)
        for t in range(steps):
            logits, state = lm.step(feed[:, start + t], state)
        return logits, state

    def test_merged_groups_match_separate_generation(self):
        lm = self._lm()
        feed_a = torch.randint(0, 64, (2, 12))
        feed_b = torch.randint(0, 64, (1, 12))
        pa = self._prompt(lm, 2, 11, 3)  # 8 real tokens, 3 left pads
        pb = self._prompt(lm, 1, 6, 0)  # 6 real tokens, none

        with torch.no_grad():
            _, sa = self._run(lm, pa, 5, feed_a)
            _, sb = self._run(lm, pb, 2, feed_b)
            separate = []
            for t in range(3):
                la, sa = lm.step(feed_a[:, 5 + t], sa)
                lb, sb = lm.step(feed_b[:, 2 + t], sb)
                separate.append(torch.cat([la, lb], dim=0))

            _, sa = self._run(lm, pa, 5, feed_a)
            _, sb = self._run(lm, pb, 2, feed_b)
            assert lm.can_merge(sa, sb)
            state = lm.merge(sa, sb)
            assert state["kv"].starts.tolist() == [3, 3, 0]
            assert state["kv"].lens_host == [16, 16, 8]
            for t, want in enumerate(separate):
                toks = torch.cat([feed_a[:, 5 + t], feed_b[:, 2 + t]])
                got, state = lm.step(toks, state)
                torch.testing.assert_close(want, got, atol=1e-5, rtol=1e-4)

    def test_positions_survive_the_merge(self):
        """Rotary positions are ``lens - starts``: the left pad must not count."""
        lm = self._lm()
        feed = torch.randint(0, 64, (1, 8))
        with torch.no_grad():
            _, sa = self._run(lm, self._prompt(lm, 1, 11, 3), 4, feed)
            _, sb = self._run(lm, self._prompt(lm, 1, 6, 0), 1, feed)
            merged = lm.merge(sa, sb)
        # 8 prompt tokens + 4 generated, and 6 + 1 -- both counted from the first
        # *real* token, not from the buffer's index 0.
        assert merged["kv"].positions().tolist() == [12, 7]

    def test_a_cat_grown_state_declares_itself_unmergeable(self):
        lm = self._lm()
        with torch.no_grad():
            _, grown = lm.prefill(*self._prompt(lm, 1, 6, 0))
            _, sized = lm.prefill(*self._prompt(lm, 1, 6, 0), capacity=48)
        assert not lm.can_merge(grown, sized)


@pytest.mark.cuda
class TestStepGraphs:
    """A decoder step replayed from a CUDA graph must be the *same* step.

    The gate is exactness, not closeness: a captured graph that picks a different
    kernel, reads a stale address, or masks a different key window returns
    plausible logits, and the only thing that catches it is comparing them
    against the eager path token for token.
    """

    @staticmethod
    def _lm(head_dim: int = 64):
        from oasr.models.speech_llm.config import SpeechLlmModelConfig
        from oasr.models.speech_llm.llm import Qwen2Lm

        torch.manual_seed(3)
        cfg = SpeechLlmModelConfig(
            vocab_size=128,
            text_hidden_size=2 * head_dim,
            text_num_attention_heads=2,
            text_num_key_value_heads=2,
            text_num_hidden_layers=2,
            text_intermediate_size=192,
        )
        return Qwen2Lm(cfg).cuda().to(torch.float16).eval()

    @staticmethod
    def _manager(lm, block_tokens=16):
        from oasr.cache.decoder_kv import DecoderKVCacheManager

        cfg = lm._cfg
        return DecoderKVCacheManager(
            DecoderKVCacheManager.build_pool(
                num_layers=cfg.text_num_hidden_layers,
                n_kv_head=cfg.text_num_key_value_heads,
                head_dim=cfg.text_head_dim,
                block_tokens=block_tokens,
                max_num_blocks=128,
                device=torch.device("cuda"),
                dtype=torch.float16,
            )
        )

    def _generate(self, lm, mgr, graphs, steps=24):
        B, P = 2, 20
        torch.manual_seed(5)
        ids = torch.randint(0, 128, (B, P), device="cuda")
        valid = torch.ones(B, P, dtype=torch.bool, device="cuda")
        valid[1, :4] = False
        feed = torch.randint(0, 128, (B, steps), device="cuda")
        with torch.no_grad():
            logits, state = lm.prefill(lm.embed_tokens(ids), valid, capacity=64, kv_manager=mgr)
            out = [logits.float().cpu()]
            for t in range(steps):
                if graphs is not None:
                    replayed = graphs.step(feed[:, t], state["kv"])
                    assert replayed is not None, "step was not captured"
                    state["kv"].commit(1)
                    logits = replayed
                else:
                    logits, state = lm.step(feed[:, t], state)
                out.append(logits.float().cpu())
        state["kv"].free()
        return out

    @pytest.mark.parametrize("head_dim,block_tokens", [(32, 16), (64, 16), (128, 16), (64, 8)])
    def test_replayed_steps_match_eager_exactly(self, head_dim, block_tokens):
        """Across head dims, because that is what a second checkpoint changes.

        The only real speech-LLM this is measured on is Qwen2-Audio-7B, whose
        decoder is `head_dim 128`.  What a different checkpoint would vary is not
        the weights — a graph does not care — but the *geometry* the paged loader
        runs at, and this repo's two capture defects were both in that loader and
        both depended on `head_dim` and `block_size` together: the shared-memory
        write-after-write race at `head_dim 32`, and the block-table width that
        must be a whole number of K tiles.  32 is the one that has bitten, and an
        8-token page is the one that puts more than four pages in a K tile.
        """
        from oasr.engine.decoder_graph import DecoderStepGraphCache

        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        lm = self._lm(head_dim)
        eager = self._generate(lm, self._manager(lm, block_tokens), None)
        mgr = self._manager(lm, block_tokens)
        graphs = DecoderStepGraphCache(lm, mgr, width_pages=2)
        replayed = self._generate(lm, mgr, graphs)
        assert graphs.num_captured >= 2  # the table widened mid-run
        for a, b in zip(eager, replayed):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def _interleaved(self, lm, mgr, graphs, steps=24):
        """Two live states of *different* row counts, stepped alternately.

        Rows are the graph cache's exact shape key, so the two states capture
        two keys, and every step of the first after the second's capture is a
        replay that happened *after* a new capture in the same memory pool.
        """
        torch.manual_seed(11)
        states, feeds = [], []
        for rows, prompt in ((2, 20), (1, 12)):
            ids = torch.randint(0, 128, (rows, prompt), device="cuda")
            valid = torch.ones(rows, prompt, dtype=torch.bool, device="cuda")
            with torch.no_grad():
                _, state = lm.prefill(lm.embed_tokens(ids), valid, capacity=64, kv_manager=mgr)
            states.append(state)
            feeds.append(torch.randint(0, 128, (rows, steps), device="cuda"))

        out = []
        with torch.no_grad():
            for t in range(steps):
                for i, state in enumerate(states):
                    if graphs is not None:
                        logits = graphs.step(feeds[i][:, t], state["kv"])
                        assert logits is not None, "step was not captured"
                        state["kv"].commit(1)
                    else:
                        logits, state = lm.step(feeds[i][:, t], state)
                        states[i] = state
                    out.append(logits.float().cpu())
        for state in states:
            state["kv"].free()
        return out

    def test_a_later_capture_does_not_invalidate_an_earlier_shape(self):
        """The rule everyone remembers is that a later *replay* invalidates the
        buffer.  The one that cost real transcripts is that a later **capture**
        does too: captures share one memory pool, so a capture at a new key can
        be handed the block an earlier capture's output buffer occupies.  In the
        streaming encoder's cache that deterministically lost 5 of 40 LJSpeech
        utterances their trailing words, graphs-only — see the
        ``nemotron_streaming`` comment in ``ci/wer-reference.json``.

        Two things keep it from happening here, and this drives both: the cache
        holds every capture's output buffer for its own lifetime, so the
        allocator cannot hand that block to the next capture, and :meth:`step`
        clones before returning.  Interleaving two row counts means each state
        replays *after* the other's capture.
        """
        from oasr.engine.decoder_graph import DecoderStepGraphCache

        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        lm = self._lm()
        eager = self._interleaved(lm, self._manager(lm), None)
        mgr = self._manager(lm)
        graphs = DecoderStepGraphCache(lm, mgr, width_pages=2)
        replayed = self._interleaved(lm, mgr, graphs)
        # Two row counts, and the table widens under a 2-page bucket: more keys
        # than states, which is the point — the later captures land mid-run.
        assert graphs.num_captured > 2
        for a, b in zip(eager, replayed):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_the_capture_ceiling_falls_back_to_eager(self):
        """Rows are an *exact* key, so shapes are not bounded by anything else.

        A pool whose row count changes every tick reaches many keys, and each
        one costs graph memory that is never reclaimed while the engine lives.
        Past ``step_graph_max_captures`` a step has to run eager rather than keep
        capturing — a ceiling that silently kept allocating would be worse than
        no ceiling, because it would look like it was holding.
        """
        from oasr.engine.decoder_graph import DecoderStepGraphCache

        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        lm = self._lm()
        mgr = self._manager(lm)
        graphs = DecoderStepGraphCache(lm, mgr, width_pages=2, max_captures=1)
        eager = self._interleaved(lm, self._manager(lm), None)
        mixed = self._interleaved_allowing_eager(lm, mgr, graphs)
        assert graphs.num_captured == 1
        # Whatever ran eager must still agree with the eager reference exactly:
        # the fallback is a different code path, not a different answer.
        for a, b in zip(eager, mixed):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def _interleaved_allowing_eager(self, lm, mgr, graphs, steps=24):
        """:meth:`_interleaved` without the "must have been captured" assert."""
        torch.manual_seed(11)
        states, feeds = [], []
        for rows, prompt in ((2, 20), (1, 12)):
            ids = torch.randint(0, 128, (rows, prompt), device="cuda")
            valid = torch.ones(rows, prompt, dtype=torch.bool, device="cuda")
            with torch.no_grad():
                _, state = lm.prefill(lm.embed_tokens(ids), valid, capacity=64, kv_manager=mgr)
            states.append(state)
            feeds.append(torch.randint(0, 128, (rows, steps), device="cuda"))
        out = []
        with torch.no_grad():
            for t in range(steps):
                for i, state in enumerate(states):
                    logits = graphs.step(feeds[i][:, t], state["kv"])
                    if logits is None:
                        logits, state = lm.step(feeds[i][:, t], state)
                        states[i] = state
                    else:
                        state["kv"].commit(1)
                    out.append(logits.float().cpu())
        for state in states:
            state["kv"].free()
        return out

    def test_a_failed_capture_is_not_retried(self):
        """Retrying a capture that already failed is the slowest outcome.

        A capture costs a warm-up forward before it records anything, so a shape
        that raises once would pay that on *every* step of that shape and then
        run eager regardless — strictly worse than never having tried.  With
        graphs on by default this is the difference between a bad configuration
        being slightly slower and being unusable.
        """
        from oasr.engine.decoder_graph import DecoderStepGraphCache

        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        lm = self._lm()
        mgr = self._manager(lm)
        graphs = DecoderStepGraphCache(lm, mgr)
        with torch.no_grad():
            ids = torch.randint(0, 128, (1, 8), device="cuda")
            valid = torch.ones(1, 8, dtype=torch.bool, device="cuda")
            _, state = lm.prefill(lm.embed_tokens(ids), valid, capacity=32, kv_manager=mgr)

        calls = {"n": 0}
        real_step = lm.step

        def exploding(tokens, st):
            calls["n"] += 1
            raise RuntimeError("capture is not going to work today")

        lm.step = exploding
        try:
            token = torch.zeros(1, dtype=torch.long, device="cuda")
            assert graphs.step(token, state["kv"]) is None
            assert calls["n"] == 1
            assert graphs.step(token, state["kv"]) is None
            assert calls["n"] == 1, "the failed shape was attempted a second time"
        finally:
            lm.step = real_step
            state["kv"].free()

    def test_dense_storage_is_refused(self):
        """A capacity buffer is allocated per group and moves with every prefill,
        so there is no address for a graph to have captured."""
        from oasr.engine.decoder_graph import DecoderStepGraphCache

        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        lm = self._lm()
        mgr = self._manager(lm)
        graphs = DecoderStepGraphCache(lm, mgr)
        with torch.no_grad():
            ids = torch.randint(0, 128, (1, 8), device="cuda")
            valid = torch.ones(1, 8, dtype=torch.bool, device="cuda")
            _, dense = lm.prefill(lm.embed_tokens(ids), valid, capacity=32)
        assert not graphs.capturable(dense["kv"])
        assert graphs.step(torch.zeros(1, dtype=torch.long, device="cuda"), dense["kv"]) is None

    def test_a_decoder_that_does_not_declare_it_is_refused(self):
        from oasr.engine.decoder_graph import DecoderStepGraphCache

        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        lm = self._lm()
        mgr = self._manager(lm)
        graphs = DecoderStepGraphCache(lm, mgr)
        with torch.no_grad():
            ids = torch.randint(0, 128, (1, 8), device="cuda")
            valid = torch.ones(1, 8, dtype=torch.bool, device="cuda")
            _, state = lm.prefill(lm.embed_tokens(ids), valid, capacity=32, kv_manager=mgr)
        assert graphs.capturable(state["kv"])
        graphs._decoder = object()  # a surface with no supports_step_graphs
        assert not graphs.capturable(state["kv"])
        state["kv"].free()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestDecoderCacheIsKernelSafe:
    """The capacity-preallocated KV cache is handed to the attention kernel
    *whole* (buffer + length), which is what lets it skip a per-layer copy of a
    stride-gapped slice.  That makes the buffer's untouched tail readable by the
    kernel's last partial K block, so it must be zeroed: uninitialized memory
    there can hold NaN, and a NaN in ``v`` survives any mask through ``P @ V``.

    The symptom is not a wrong answer on one run — it is *different* answers on
    repeated identical runs, which parity tests cannot see because each run
    looks plausible on its own.
    """

    def _lm(self):
        from oasr.models.speech_llm.config import SpeechLlmModelConfig
        from oasr.models.speech_llm.llm import Qwen2Lm

        torch.manual_seed(0)
        cfg = SpeechLlmModelConfig(
            vocab_size=256,
            text_hidden_size=4 * 64,
            text_num_attention_heads=4,
            text_num_key_value_heads=4,
            text_num_hidden_layers=2,
            text_intermediate_size=256,
        )
        return Qwen2Lm(cfg).cuda().to(torch.float16).eval()

    def test_preallocated_cache_tail_is_zeroed(self):
        lm = self._lm()
        B, P = 2, 96
        with torch.inference_mode():
            emb = lm.embed_tokens(torch.randint(0, 256, (B, P), device="cuda"))
            valid = torch.ones(B, P, dtype=torch.bool, device="cuda")
            _, state = lm.prefill(emb, valid, capacity=P + 40)
        kv = state["kv"]
        for buf in list(kv.k) + list(kv.v):
            tail = buf[:, :, P:]
            assert torch.isfinite(buf).all(), "cache holds non-finite values"
            torch.testing.assert_close(tail, torch.zeros_like(tail))

    def test_repeated_runs_agree(self):
        """Determinism, which is what the uninitialized tail actually broke."""
        lm = self._lm()
        B, P = 2, 96
        outs = []
        for _ in range(4):
            with torch.inference_mode():
                emb = lm.embed_tokens(torch.arange(B * P, device="cuda").remainder(256).view(B, P))
                valid = torch.ones(B, P, dtype=torch.bool, device="cuda")
                valid[1, :16] = False
                logits, state = lm.prefill(emb, valid, capacity=P + 8)
                toks = [logits.argmax(-1)]
                for _ in range(4):
                    logits, state = lm.step(toks[-1], state)
                    toks.append(logits.argmax(-1))
            outs.append(torch.stack(toks))
        for i, o in enumerate(outs[1:], 1):
            assert torch.equal(o, outs[0]), f"run {i} disagrees with run 0"

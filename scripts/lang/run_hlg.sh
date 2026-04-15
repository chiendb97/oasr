#!/usr/bin/env bash
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end pipeline for building an HLG decoding graph.
#
# Supports three token types (--token-type):
#   bpe   — SentencePiece BPE/unigram model (requires --bpe-model, --units-file)
#   char  — Character-level tokens           (requires lang-dir/text and lang-dir/words.txt)
#   phone — Phone-level tokens               (requires lang-dir/lexicon.txt)
#
# Stages:
#   1  prepare_words   Build words.txt from training text  [bpe only]
#   2  prepare_lang    Build tokens.txt, lexicon, L.pt
#   3  make_kn_lm      Train Kneser-Ney n-gram LM
#   4  kaldilm         Convert ARPA → OpenFst text format
#   5  compile_hlg     Compile HLG.pt
#
# Usage (BPE):
#   bash scripts/run_hlg.sh \
#       --token-type bpe \
#       --am-dir /path/to/am \
#       --text   corpus/train-100/text corpus/train-360/text corpus/train-500/text \
#       --lang-dir data/lang_bpe \
#       --lm-dir   data/lm
#
# Usage (char):
#   bash scripts/run_hlg.sh \
#       --token-type char \
#       --lang-dir data/lang_char \
#       --lm-dir   data/lm
#
# Usage (phone):
#   bash scripts/run_hlg.sh \
#       --token-type phone \
#       --lang-dir data/lang_phone \
#       --lm-dir   data/lm \
#       --text     corpus/text
#
# Resume from a specific stage:
#   bash scripts/run_hlg.sh ... --stage 3
#
# Stop after a specific stage:
#   bash scripts/run_hlg.sh ... --stop-stage 3

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
token_type=bpe
am_dir=
bpe_model=          # defaults to $am_dir/train_960_unigram5000.model if empty
units_file=         # defaults to $am_dir/units.txt if empty
text=()             # one or more plain-text corpus files
lang_dir=data/lang_bpe
lm_dir=data/lm
lm_stem=G_3_gram
ngram_order=3
min_word_count=1
sil_token=SIL
sil_prob=0.5
stage=1
stop_stage=5
overwrite=false

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
    sed -n '3,/^set -euo/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'
    exit 0
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --token-type)    token_type="$2";    shift 2 ;;
        --am-dir)        am_dir="$2";        shift 2 ;;
        --bpe-model)     bpe_model="$2";     shift 2 ;;
        --units-file)    units_file="$2";    shift 2 ;;
        --lang-dir)      lang_dir="$2";      shift 2 ;;
        --lm-dir)        lm_dir="$2";        shift 2 ;;
        --lm-stem)       lm_stem="$2";       shift 2 ;;
        --ngram-order)   ngram_order="$2";   shift 2 ;;
        --min-word-count) min_word_count="$2"; shift 2 ;;
        --sil-token)     sil_token="$2";     shift 2 ;;
        --sil-prob)      sil_prob="$2";      shift 2 ;;
        --stage)         stage="$2";         shift 2 ;;
        --stop-stage)    stop_stage="$2";    shift 2 ;;
        --overwrite)     overwrite=true;     shift   ;;
        --text)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                text+=("$1"); shift
            done
            ;;
        -h|--help) usage ;;
        *) die "Unknown option: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate token type
# ---------------------------------------------------------------------------
case "$token_type" in
    bpe|char|phone) ;;
    *) die "--token-type must be one of: bpe, char, phone (got: '$token_type')" ;;
esac

# Resolve BPE model / units file paths from am_dir if not set explicitly.
if [[ "$token_type" == "bpe" ]]; then
    [[ -n "$am_dir" || ( -n "$bpe_model" && -n "$units_file" ) ]] || \
        die "BPE mode requires --am-dir, or both --bpe-model and --units-file."
    [[ -z "$bpe_model"   ]] && bpe_model="${am_dir}/train_960_unigram5000.model"
    [[ -z "$units_file"  ]] && units_file="${am_dir}/units.txt"
fi

# At least one text file is required except for char (which reads from lang_dir/text).
if [[ "$token_type" != "char" && ${#text[@]} -eq 0 ]]; then
    die "--text FILE [FILE ...] is required for token-type '$token_type'."
fi

overwrite_flag=""
[[ "$overwrite" == "true" ]] && overwrite_flag="--overwrite"

mkdir -p "$lang_dir" "$lm_dir"

log "=== HLG pipeline: token_type=$token_type, stages $stage–$stop_stage ==="
log "    lang_dir : $lang_dir"
log "    lm_dir   : $lm_dir"

# ---------------------------------------------------------------------------
# Stage 1 — Build words.txt  [bpe only]
# ---------------------------------------------------------------------------
if [[ $stage -le 1 && $stop_stage -ge 1 ]]; then
    if [[ "$token_type" == "bpe" ]]; then
        words_out="${lang_dir}/words.txt"
        if [[ -f "$words_out" && "$overwrite" == "false" ]]; then
            log "Stage 1: $words_out already exists — skipping (use --overwrite to force)"
        else
            log "Stage 1: Building words.txt from ${#text[@]} text file(s) ..."
            python3 "${SCRIPT_DIR}/prepare_words.py" \
                --text "${text[@]}" \
                --lang-dir "$lang_dir" \
                --min-count "$min_word_count"
            log "Stage 1: Done → $words_out"
        fi
    else
        log "Stage 1: Skipped (not needed for token-type '$token_type')"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 2 — Prepare lang directory (tokens.txt, lexicon, L.pt)
# ---------------------------------------------------------------------------
if [[ $stage -le 2 && $stop_stage -ge 2 ]]; then
    l_out="${lang_dir}/L_disambig.pt"
    if [[ -f "$l_out" && "$overwrite" == "false" ]]; then
        log "Stage 2: $l_out already exists — skipping (use --overwrite to force)"
    else
        log "Stage 2: Preparing lang directory (token-type=$token_type) ..."
        case "$token_type" in
            bpe)
                python3 "${SCRIPT_DIR}/prepare_bpe.py" \
                    --lang-dir   "$lang_dir" \
                    --bpe-model  "$bpe_model" \
                    --units-file "$units_file"
                ;;
            char)
                python3 "${SCRIPT_DIR}/prepare_char.py" \
                    --lang-dir "$lang_dir"
                ;;
            phone)
                python3 "${SCRIPT_DIR}/prepare_lang.py" \
                    --lang-dir  "$lang_dir" \
                    --sil-token "$sil_token" \
                    --sil-prob  "$sil_prob"
                ;;
        esac
        log "Stage 2: Done → $lang_dir"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 3 — Train n-gram language model
# ---------------------------------------------------------------------------
if [[ $stage -le 3 && $stop_stage -ge 3 ]]; then
    arpa_out="${lm_dir}/${lm_stem}.arpa"
    if [[ -f "$arpa_out" && "$overwrite" == "false" ]]; then
        log "Stage 3: $arpa_out already exists — skipping (use --overwrite to force)"
    else
        # For char, use lang_dir/text as the default corpus if --text was not given.
        lm_text_args=()
        if [[ ${#text[@]} -gt 0 ]]; then
            lm_text_args=(-text "${text[@]}")
        elif [[ "$token_type" == "char" ]]; then
            [[ -f "${lang_dir}/text" ]] || \
                die "Stage 3: no --text files given and ${lang_dir}/text not found."
            lm_text_args=(-text "${lang_dir}/text")
        else
            die "Stage 3: --text FILE [FILE ...] required."
        fi

        log "Stage 3: Training ${ngram_order}-gram LM ..."
        python3 "${SCRIPT_DIR}/make_kn_lm.py" \
            -ngram-order "$ngram_order" \
            "${lm_text_args[@]}" \
            -lm "$arpa_out"
        log "Stage 3: Done → $arpa_out"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 4 — Convert ARPA → OpenFst text format (kaldilm)
# ---------------------------------------------------------------------------
if [[ $stage -le 4 && $stop_stage -ge 4 ]]; then
    fst_txt_out="${lm_dir}/${lm_stem}.fst.txt"
    if [[ -f "$fst_txt_out" && "$overwrite" == "false" ]]; then
        log "Stage 4: $fst_txt_out already exists — skipping (use --overwrite to force)"
    else
        arpa_in="${lm_dir}/${lm_stem}.arpa"
        [[ -f "$arpa_in" ]] || die "Stage 4: ARPA file not found: $arpa_in"

        python3 -c "import kaldilm" 2>/dev/null || \
            die "Stage 4: kaldilm not found. Install with: pip install kaldilm"

        log "Stage 4: Converting ARPA to OpenFst text format ..."
        python3 -m kaldilm \
            --read-symbol-table="${lang_dir}/words.txt" \
            --disambig-symbol='#0' \
            --max-order="$ngram_order" \
            "$arpa_in" > "$fst_txt_out"
        log "Stage 4: Done → $fst_txt_out"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 5 — Compile HLG
# ---------------------------------------------------------------------------
if [[ $stage -le 5 && $stop_stage -ge 5 ]]; then
    hlg_out="${lang_dir}/HLG.pt"
    if [[ -f "$hlg_out" && "$overwrite" == "false" ]]; then
        log "Stage 5: $hlg_out already exists — skipping (use --overwrite to force)"
    else
        log "Stage 5: Compiling HLG ..."
        python3 "${SCRIPT_DIR}/compile_hlg.py" \
            --lang-dir "$lang_dir" \
            --lm       "$lm_stem" \
            --lm-dir   "$lm_dir" \
            $overwrite_flag
        log "Stage 5: Done → $hlg_out"
    fi
fi

log "=== Pipeline complete ==="
log ""
log "Load the graph with the OASR WFST decoder:"
log "  from oasr.decode import Decoder, DecoderConfig"
log "  decoder = Decoder(DecoderConfig(search_type='wfst'), fst='${lang_dir}/HLG.pt')"

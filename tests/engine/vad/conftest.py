# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""What every VAD test needs, in one place.

``tone`` and ``hiss`` were byte-identical in two of these modules (differing
only in a seed), and the "n utterances separated by silence" corpus existed
twice, once returning the span list and once not.  A VAD test is mostly its
fixture, so a forked fixture is a forked definition of what counts as speech.
They now live in ``helpers.audio`` with the other waveform builders; this file
exists so the sibling modules can say where they come from.
"""

from helpers.audio import SR, hiss, speech_corpus, tone

__all__ = ["SR", "hiss", "speech_corpus", "tone"]

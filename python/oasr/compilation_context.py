# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CUDA compilation context for managing target architectures.

Detects available GPU architectures and generates NVCC flags for JIT
compilation. Supports manual override via ``OASR_CUDA_ARCH_LIST``
environment variable.
"""

import os
import subprocess
from typing import List, Optional, Set, Tuple


class CompilationContext:
    """Manages CUDA architecture targets for JIT compilation.

    On construction, detects all GPUs in the system (or reads from
    ``OASR_CUDA_ARCH_LIST`` env var). For SM >= 90, the ``a`` suffix
    is added automatically (e.g., ``9.0a`` for Hopper).

    Parameters
    ----------
    cuda_arch_list : str, optional
        Explicit architecture list (e.g., ``"8.0 9.0a 10.0a"``).
        If None, reads from ``OASR_CUDA_ARCH_LIST`` env var or
        auto-detects from available GPUs.
    """

    def __init__(self, cuda_arch_list: Optional[str] = None):
        if cuda_arch_list is None:
            cuda_arch_list = os.environ.get("OASR_CUDA_ARCH_LIST", "")

        if cuda_arch_list.strip():
            self.TARGET_CUDA_ARCHS = self._parse_arch_list(cuda_arch_list)
        else:
            self.TARGET_CUDA_ARCHS = self._detect_archs()

    @staticmethod
    def _parse_arch_list(arch_str: str) -> Set[Tuple[int, str]]:
        """Parse an architecture list string like ``"8.0 9.0a 10.0a"``."""
        archs = set()
        for token in arch_str.strip().split():
            token = token.strip().rstrip(",")
            if not token:
                continue
            # Handle suffix (e.g., "9.0a")
            suffix = ""
            if token[-1].isalpha():
                suffix = token[-1]
                token = token[:-1]
            parts = token.split(".")
            major = int(parts[0])
            minor = parts[1] if len(parts) > 1 else "0"
            archs.add((major, minor + suffix))
        return archs

    @staticmethod
    def _detect_archs() -> Set[Tuple[int, str]]:
        """Auto-detect GPU architectures from the system."""
        archs = set()
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    major, minor = props.major, props.minor
                    suffix = "a" if major >= 9 else ""
                    archs.add((major, f"{minor}{suffix}"))
                return archs
        except ImportError:
            pass

        # Fallback: nvidia-smi
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                text=True,
            )
            for line in out.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(".")
                major = int(parts[0])
                minor = parts[1] if len(parts) > 1 else "0"
                suffix = "a" if major >= 9 else ""
                archs.add((major, f"{minor}{suffix}"))
        except Exception:
            # Safe default: SM80
            archs.add((8, "0"))

        return archs

    def get_nvcc_flags_list(
        self, supported_major_versions: Optional[List[int]] = None
    ) -> List[str]:
        """Generate NVCC architecture flags for the target architectures.

        Parameters
        ----------
        supported_major_versions : list of int, optional
            Major SM versions that the kernel supports (e.g., ``[9, 10, 11, 12]``
            for Hopper+). If ``None``, all detected architectures are used.

        Returns
        -------
        list of str
            NVCC flags like ``["-gencode=arch=compute_90a,code=sm_90a", ...]``.

        Raises
        ------
        RuntimeError
            If no target architectures match the supported versions.
        """
        flags = []
        for major, minor_suffix in sorted(self.TARGET_CUDA_ARCHS):
            if supported_major_versions is not None:
                if major not in supported_major_versions:
                    continue
            sm = f"{major}{minor_suffix}"
            flags.append(f"-gencode=arch=compute_{sm},code=sm_{sm}")

        if not flags:
            if supported_major_versions is not None:
                raise RuntimeError(
                    f"No supported CUDA architectures found for major versions "
                    f"{supported_major_versions}. Available: {self.TARGET_CUDA_ARCHS}"
                )
            raise RuntimeError(
                "No CUDA architectures detected. Set OASR_CUDA_ARCH_LIST "
                "or ensure a CUDA-capable GPU is available."
            )

        return flags

    def get_sm_list(self) -> List[int]:
        """Return sorted list of SM versions as integers (e.g., [80, 90])."""
        sms = []
        for major, minor_suffix in self.TARGET_CUDA_ARCHS:
            minor_digit = ""
            for ch in minor_suffix:
                if ch.isdigit():
                    minor_digit += ch
                else:
                    break
            sm = major * 10 + int(minor_digit or "0")
            sms.append(sm)
        return sorted(set(sms))

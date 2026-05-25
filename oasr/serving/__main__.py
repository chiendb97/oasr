# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``python -m oasr.serving`` — run one engine worker."""

import sys

from .engine_worker import main

if __name__ == "__main__":
    sys.exit(main())

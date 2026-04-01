# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Jinja2 template rendering utilities for JIT code generation."""

import jinja2

from . import env


def get_template_env() -> jinja2.Environment:
    """Create a Jinja2 environment loading templates from csrc/templates/."""
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(env.OASR_TEMPLATE_DIR)),
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
    )


def render_template(template_name: str, **kwargs) -> str:
    """Render a Jinja2 template with the given variables.

    Args:
        template_name: Name of the template file (e.g., "gemm_cutlass_template.cu.jinja").
        **kwargs: Template variables.

    Returns:
        Rendered source code as a string.
    """
    template_env = get_template_env()
    tmpl = template_env.get_template(template_name)
    return tmpl.render(**kwargs)

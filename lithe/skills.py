"""Deprecated alias: skill packages moved to :mod:`lithe.bundles.skills`.

Skills do file (and, with remote sources, network) I/O, so they belong with
the optional bundles — ``import lithe`` alone stays zero-I/O. This module
remains importable for compatibility with older host code; new code should
import from ``lithe.bundles`` (or ``lithe.bundles.skills``) directly.

Importing it emits a :class:`DeprecationWarning` (visible under
``python -W error::DeprecationWarning`` / pytest's ``filterwarnings=error``)
so hosts notice the migration instead of silently depending on the alias.
"""
from __future__ import annotations

import warnings

from lithe.bundles.skills import (  # noqa: F401
    PACKAGE_FILE, ROOT_PACKAGE, SKILL_MAIN, RemoteSkillSource, SkillPackages,
)

__all__ = ["PACKAGE_FILE", "ROOT_PACKAGE", "SKILL_MAIN",
           "RemoteSkillSource", "SkillPackages"]

warnings.warn(
    "lithe.skills is deprecated; import from lithe.bundles "
    "(or lithe.bundles.skills) instead",
    DeprecationWarning,
    stacklevel=2,
)

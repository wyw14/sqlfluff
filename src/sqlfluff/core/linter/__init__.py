"""Linter class and helper classes."""

from sqlfluff.core.formatter import FormatterInterface
from sqlfluff.core.linter.common import ParsedString, RenderedFile, RuleTuple
from sqlfluff.core.linter.conflict import (
    ConflictingEditsError,
    EditRelation,
    FixConflictPreview,
    PatchSuggestion,
    SourceEditPreview,
    StaleSourceError,
    apply_source_patches,
    build_fix_preview,
)
from sqlfluff.core.linter.linted_file import LintedFile
from sqlfluff.core.linter.linter import Linter
from sqlfluff.core.linter.linting_result import LintingResult

__all__ = (
    "ConflictingEditsError",
    "EditRelation",
    "FormatterInterface",
    "FixConflictPreview",
    "LintedFile",
    "LintingResult",
    "Linter",
    "ParsedString",
    "PatchSuggestion",
    "RenderedFile",
    "RuleTuple",
    "SourceEditPreview",
    "StaleSourceError",
    "apply_source_patches",
    "build_fix_preview",
)

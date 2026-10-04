"""Linter class and helper classes."""

from sqlfluff.core.formatter import FormatterInterface
from sqlfluff.core.linter.common import ParsedString, RenderedFile, RuleTuple
from sqlfluff.core.linter.linted_file import LintedFile
from sqlfluff.core.linter.linter import Linter
from sqlfluff.core.linter.linting_result import LintingResult
from sqlfluff.core.linter.statement_fixes import (
    KIND_BOUNDARY,
    KIND_BOUNDARY_TEMPLATE,
    KIND_FILE,
    KIND_STATEMENT,
    KIND_TEMPLATE,
    StaleSourceError,
    StatementEdit,
    StatementFixGroup,
    StatementFixPreview,
    StatementFixResult,
    apply_statement_fixes,
    build_statement_fix_preview,
    render_fixed_source,
)

__all__ = (
    "KIND_BOUNDARY",
    "KIND_BOUNDARY_TEMPLATE",
    "KIND_FILE",
    "KIND_STATEMENT",
    "KIND_TEMPLATE",
    "FormatterInterface",
    "RuleTuple",
    "ParsedString",
    "LintedFile",
    "Linter",
    "LintingResult",
    "RenderedFile",
    "StaleSourceError",
    "StatementEdit",
    "StatementFixGroup",
    "StatementFixPreview",
    "StatementFixResult",
    "apply_statement_fixes",
    "build_statement_fix_preview",
    "render_fixed_source",
)

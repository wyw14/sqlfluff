"""Statement-level fix preview and selective application.

When a source file contains more than one SQL statement, a reviewer may want
to accept fixes for only some of them. This module provides:

* :func:`build_statement_fix_preview`, which breaks the total set of fixes
  for a :class:`~sqlfluff.core.linter.linted_file.LintedFile` down into
  per-statement edits, making the ownership of every edit explicit.
* :func:`render_fixed_source` / :func:`apply_statement_fixes`, which render
  and (optionally) persist the source for a chosen subset of statements.

Edit ownership rules (these are deliberately conservative):

* An edit which lies entirely within one source statement is owned by that
  statement and is applied when it is selected.
* An edit on the boundary between two statements (for example whitespace
  inserted between two adjacent semicolon-terminated statements) is owned by
  both statements and is only applied when both are selected.
* An edit in a shared template fragment (for example literal source inside a
  Jinja loop which renders into several statements) is owned by every
  statement produced by that fragment and is only applied when all of them
  are selected.
* Edits outside of any statement (leading/trailing file padding) are only
  applied when all statements are selected.

Edits which are not selected are left byte-for-byte untouched. After a
selection is rendered, the caller must re-check that the result still parses.
Finally, applying a preview is only allowed against the exact source text the
preview was built from: if the source has changed in the meantime, the
application is rejected rather than overwriting new content at stale offsets.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import regex

from sqlfluff.core.errors import (
    SQLBaseError,
    SQLLexError,
    SQLLintError,
    SQLParseError,
    SQLTemplaterError,
)
from sqlfluff.core.templaters import TemplatedFile

if TYPE_CHECKING:  # pragma: no cover
    from sqlfluff.core.linter.linted_file import LintedFile
    from sqlfluff.core.parser.segments import BaseSegment

linter_logger = logging.getLogger("sqlfluff.linter")

# Edit kinds.
#: An edit owned by a single statement.
KIND_STATEMENT = "statement"
#: An edit on the boundary of two adjacent statements.
KIND_BOUNDARY = "boundary"
#: An edit in a template fragment which renders into multiple statements.
KIND_TEMPLATE = "template"
#: An edit which is both on a boundary and in a shared template fragment.
KIND_BOUNDARY_TEMPLATE = "boundary_and_template"
#: An edit outside any statement (leading/trailing file padding).
KIND_FILE = "file"

_SHARED_KINDS = frozenset(
    {KIND_BOUNDARY, KIND_TEMPLATE, KIND_BOUNDARY_TEMPLATE, KIND_FILE}
)

# Tokenise into runs of word characters, runs of whitespace or individual
# punctuation characters. Token level alignment (rather than character level)
# keeps independent edits from being merged across short equal spans.
_TOKEN_RE = regex.compile(r"\w+|\s+|[^\w\s]")
_NEWLINE_RE = regex.compile(r"\r\n|\r")

# Parse/template/lex failures which make a selectively fixed result invalid.
_REPARSE_FAILURE_TYPES = (SQLLexError, SQLParseError, SQLTemplaterError)


class StaleSourceError(ValueError):
    """Raised when a fix preview is applied against changed source.

    The source offsets of a preview are only valid for the exact source text
    the preview was built from. If the current source (re-read from disk or
    supplied by the caller) differs, application is refused so that stale
    offsets can never overwrite newer content.
    """

    def __init__(self, path: str, expected_hash: str, actual_hash: str) -> None:
        self.path = path
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        super().__init__(
            f"Refusing to apply the statement fix preview for {path!r}: the "
            "source file has changed since the preview was generated. Re-run "
            "the lint/fix review against the current content instead of "
            "applying edits computed against the old content."
        )


@dataclass(frozen=True)
class StatementEdit:
    """A single atomic source edit with explicit statement ownership.

    Attributes:
        kind: One of the ``KIND_*`` constants.
        statement_indices: Statements which own the edit. A ``statement``
            edit has exactly one owner; ``boundary``/``template`` edits have
            all jointly-required owners; ``file`` edits have every statement
            as owner (and are therefore held until everything is selected).
        source_slice: Slice of the original source string which the edit
            replaces (may be zero length for pure insertions).
        original_source: The source text covered by ``source_slice``.
        fixed_source: What the edit replaces it with.
        rule_codes: Rule codes associated with the edit, best effort.
        reason: Human-readable explanation of the ownership decision.
    """

    kind: str
    statement_indices: tuple[int, ...]
    source_slice: slice
    original_source: str
    fixed_source: str
    rule_codes: tuple[str, ...] = ()
    reason: str = ""

    @property
    def is_shared(self) -> bool:
        """Whether this edit requires multiple statements to be selected."""
        return self.kind in _SHARED_KINDS


@dataclass(frozen=True)
class StatementFixGroup:
    """A single source statement and the edits owned by it.

    Attributes:
        index: Zero-based statement index in source order.
        source_slice: Slice of the source string covered by the statement
            (including its terminating delimiter, when present).
        line_no: One-based line number of the start of the statement.
        line_pos: One-based line position of the start of the statement.
        original_source: The original source text of the statement.
        fixed_source: How the statement looks when only its own (exclusively
            owned) edits are applied. Shared edits touching the statement are
            held back, so this may still contain issues fixed in the full run.
        rule_codes: Rule codes addressed by edits owned by the statement.
        edits: The edits exclusively owned by this statement.
    """

    index: int
    source_slice: slice
    line_no: int
    line_pos: int
    original_source: str
    fixed_source: str
    rule_codes: tuple[str, ...]
    edits: tuple[StatementEdit, ...]


@dataclass(frozen=True)
class StatementFixResult:
    """The result of applying a selection of statement fixes.

    Attributes:
        success: Whether the selected fixes were applied (and persisted, if
            requested). False when the rendered result fails the reparse
            check; in that case nothing is written.
        fixed_source: The rendered source, including any held edits left as
            the original text.
        applied_edits: Edits which were applied.
        held_edits: Edits which were left untouched because not all of their
            owning statements were selected.
        violations: Reparse violations when the result does not parse.
        path: The path the result was persisted to, otherwise None.
    """

    success: bool
    fixed_source: str
    applied_edits: tuple[StatementEdit, ...]
    held_edits: tuple[StatementEdit, ...]
    violations: tuple[SQLBaseError, ...] = ()
    path: Optional[str] = None


@dataclass
class StatementFixPreview:
    """A per-statement breakdown of the fixes available for a file.

    Build one with :func:`build_statement_fix_preview`, inspect the
    statement groups and shared/file-level edits, then apply a selection with
    :func:`render_fixed_source` or
    :func:`apply_statement_fixes`.

    Attributes:
        path: Path of the linted file (may be a generated name).
        encoding: Encoding to use when persisting the file.
        source_hash: SHA256 of the normalised source text the preview was
            built from. Used to reject application against changed source.
        statements: Statement groups in source order.
        edits: Every edit in source order, including shared/file edits.
    """

    path: str
    encoding: str
    source_hash: str
    statements: list[StatementFixGroup]
    edits: list[StatementEdit]
    _source_str: str = field(repr=False)
    _fixed_str: str = field(repr=False)
    _hunks: list["_Hunk"] = field(repr=False, default_factory=list)

    def edits_for_statement(self, index: int) -> tuple[StatementEdit, ...]:
        """Return the edits exclusively owned by statement ``index``."""
        return self.statements[index].edits

    @property
    def shared_edits(self) -> tuple[StatementEdit, ...]:
        """Edits which require multiple statements (or all) to be selected."""
        return tuple(edit for edit in self.edits if edit.is_shared)

    @property
    def has_changes(self) -> bool:
        """Whether the full fix run produced any edits at all."""
        return bool(self._hunks)

    @property
    def full_fixed_source(self) -> str:
        """The source after applying every available edit."""
        return self._fixed_str


# -----------------------------------------------------------------------------
# Token-level diffing
# -----------------------------------------------------------------------------


def _normalise_newlines(source: str) -> str:
    """Normalise newlines to unix-style line endings.

    This mirrors ``Linter._normalise_newlines``: source is normalised before
    templating, so preview offsets and hash checks use the same convention.
    """
    return _NEWLINE_RE.sub("\n", source)


def _token_spans(source: str) -> list[tuple[str, int, int]]:
    tokens = [
        (match.group(), match.start(), match.end())
        for match in _TOKEN_RE.finditer(source)
    ]
    # The token patterns must cover every character so that reconstructing
    # the source from the hunks is lossless.
    assert "".join(token[0] for token in tokens) == source
    return tokens


def _diff_hunks(original: str, fixed: str) -> list[tuple[int, int, int, int, str, str]]:
    """Return non-equal aligned hunks between ``original`` and ``fixed``.

    Each hunk is ``(orig_start, orig_stop, fixed_start, fixed_stop,
    original_text, fixed_text)`` with offsets into the respective strings.
    """
    original_tokens = _token_spans(original)
    fixed_tokens = _token_spans(fixed)
    sequence_matcher = difflib.SequenceMatcher(
        None,
        [token[0] for token in original_tokens],
        [token[0] for token in fixed_tokens],
        autojunk=False,
    )
    hunks: list[tuple[int, int, int, int, str, str]] = []
    for tag, i1, i2, j1, j2 in sequence_matcher.get_opcodes():
        if tag == "equal":
            continue
        orig_start = (
            original_tokens[i1][1] if i1 < len(original_tokens) else len(original)
        )
        orig_stop = original_tokens[i2 - 1][2] if i2 > i1 else orig_start
        fixed_start = fixed_tokens[j1][1] if j1 < len(fixed_tokens) else len(fixed)
        fixed_stop = fixed_tokens[j2 - 1][2] if j2 > j1 else fixed_start
        hunks.append(
            (
                orig_start,
                orig_stop,
                fixed_start,
                fixed_stop,
                original[orig_start:orig_stop],
                fixed[fixed_start:fixed_stop],
            )
        )
    return hunks


# -----------------------------------------------------------------------------
# Statement regions in templated space
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Region:
    """A tiled region of templated space.

    ``kind`` is ``statement``, ``gap`` (between two adjacent statements) or
    ``file`` (leading/trailing padding). ``owners`` are statement indices;
    file regions have no owners.
    """

    kind: str
    owners: tuple[int, ...]
    start: int
    stop: int


def _build_regions(
    tree: "BaseSegment", templated_file: TemplatedFile
) -> tuple[list[_Region], int]:
    """Tile the templated file into statement / gap / file regions."""
    length = len(templated_file.templated_str)
    statements = [
        segment
        for segment in tree.get_children("statement", "unparsable")
        if segment.pos_marker
    ]
    statements.sort(key=lambda seg: _templated_slice(seg).start)
    statement_count = len(statements)

    # Real (non zero length), deduplicated code segments between statements.
    # Zero length segments are inserted points (e.g. new layout whitespace)
    # or placeholders duplicated by loops; their positions are handled via
    # region tiling instead.
    seen: set[tuple[str, int, int]] = set()
    code_children = []
    for child in tree.segments:
        if (
            not child.is_code
            or not child.pos_marker
            or child.type in ("statement", "unparsable")
        ):
            continue
        child_slice = _templated_slice(child)
        if child_slice.start == child_slice.stop:
            continue
        key = (child.type, child_slice.start, child_slice.stop)
        if key in seen:
            continue
        seen.add(key)
        code_children.append(child)
    code_children.sort(key=lambda seg: _templated_slice(seg).start)

    regions: list[_Region] = []
    # Leading padding. Always emitted (possibly zero length) so point inserts
    # before the first statement are attributable to the file rather than to
    # the first statement.
    first_start = _templated_slice(statements[0]).start if statements else length
    regions.append(_Region("file", (), 0, first_start))
    cursor = first_start

    for index, statement in enumerate(statements):
        statement_slice = _templated_slice(statement)
        start = statement_slice.start
        stop = statement_slice.stop
        if index > 0 and start >= cursor:
            regions.append(_Region("gap", (index - 1, index), cursor, start))
        # Extend the statement through its terminating delimiter. The first
        # code token in the gap (usually a semicolon) ends the statement
        # region; whitespace before it is therefore owned by the statement,
        # while whitespace after it is a shared boundary gap.
        end = stop
        next_start = (
            _templated_slice(statements[index + 1]).start
            if index + 1 < statement_count
            else length
        )
        delimiter = next(
            (
                child
                for child in code_children
                if stop <= _templated_slice(child).start
                and _templated_slice(child).stop <= next_start
            ),
            None,
        )
        if delimiter is not None:
            end = _templated_slice(delimiter).stop
        regions.append(_Region("statement", (index,), start, end))
        cursor = end

    # Trailing padding, plus a zero length EOF region so point inserts at the
    # very end of the file are attributable.
    regions.append(_Region("file", (), cursor, length))
    regions.append(_Region("file", (), length, length))
    return regions, statement_count


def _templated_slice(segment: "BaseSegment") -> slice:
    """Return the templated slice of a segment known to have a marker."""
    assert segment.pos_marker
    return segment.pos_marker.templated_slice


def _point_in_region(point: int, region: _Region) -> bool:
    if region.start == region.stop:
        # Zero length regions only contain their own point.
        return point == region.start
    return region.start <= point < region.stop


def _interval_overlaps_region(start: int, stop: int, region: _Region) -> bool:
    if start == stop:
        return _point_in_region(start, region)
    return start < region.stop and stop > region.start


def _rendered_intervals(
    templated_file: TemplatedFile, source_start: int, source_stop: int
) -> list[tuple[int, int]]:
    """Map a source range to the templated intervals rendered from it.

    Literal slices map 1:1, so an exact sub-interval is returned. Non-literal
    slices (template tags/placeholders) collapse to their whole rendered
    interval, which is conservative. A literal source fragment rendered more
    than once (for example in a loop) therefore maps to multiple intervals,
    exposing shared template fragments.
    """
    point = source_start == source_stop
    intervals: list[tuple[int, int]] = []
    for file_slice in templated_file.sliced_file:
        s_start, s_stop = (
            file_slice.source_slice.start,
            file_slice.source_slice.stop,
        )
        t_start, t_stop = (
            file_slice.templated_slice.start,
            file_slice.templated_slice.stop,
        )
        if point:
            if not s_start <= source_start <= s_stop:
                continue
            if file_slice.slice_type == "literal" and s_stop > s_start:
                mapped = t_start + min(max(source_start - s_start, 0), s_stop - s_start)
                intervals.append((mapped, mapped))
            else:
                intervals.append((t_start, t_stop))
        elif s_start < source_stop and s_stop > source_start:
            if (
                file_slice.slice_type == "literal"
                and s_stop - s_start == t_stop - t_start
            ):
                low = t_start + max(source_start - s_start, 0)
                high = t_stop - max(s_stop - source_stop, 0)
                intervals.append((min(low, high), max(low, high)))
            else:
                intervals.append((t_start, t_stop))

    intervals.sort()
    merged: list[tuple[int, int]] = []
    for interval in intervals:
        if merged and interval[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
        else:
            merged.append(interval)
    return merged


def _attribute_hunk(
    source_start: int,
    source_stop: int,
    templated_file: TemplatedFile,
    regions: Sequence[_Region],
    statement_count: int,
) -> tuple[str, tuple[int, ...]]:
    """Attribute one source hunk to statements.

    Returns a ``(kind, owners)`` pair, where ``owners`` are all statements
    which must be selected for the edit to be applied.
    """
    intervals = _rendered_intervals(templated_file, source_start, source_stop)

    statement_owners: set[int] = set()
    gap_owners: set[int] = set()
    touches_file = False
    per_interval_owners: list[set[int]] = []

    for interval_start, interval_stop in intervals:
        this_interval_statements: set[int] = set()
        for region in regions:
            if not _interval_overlaps_region(interval_start, interval_stop, region):
                continue
            if region.kind == "statement":
                statement_owners.update(region.owners)
                this_interval_statements.update(region.owners)
            elif region.kind == "gap":
                gap_owners.update(region.owners)
            else:
                touches_file = True
        per_interval_owners.append(this_interval_statements)

    # File padding edits are held until every statement is selected. When the
    # file contains no statements, there is nothing they can be attributed to.
    if touches_file:
        return KIND_FILE, tuple(range(statement_count))

    # Edits entirely within a single statement are the simple case.
    if not gap_owners and len(statement_owners) == 1:
        return KIND_STATEMENT, (next(iter(statement_owners)),)

    # Anything else touches a boundary or spans multiple statements. A shared
    # template fragment is detected when the same source range renders into
    # disjoint templated intervals belonging to different statements.
    rendered_statement_sets = [owners for owners in per_interval_owners if owners]
    is_template = (
        len(rendered_statement_sets) > 1
        and len(set().union(*rendered_statement_sets)) > 1
    )
    is_boundary = bool(gap_owners) or len(statement_owners) > 1

    if is_template and is_boundary:
        kind = KIND_BOUNDARY_TEMPLATE
    elif is_template:
        kind = KIND_TEMPLATE
    else:
        kind = KIND_BOUNDARY

    owners = statement_owners | gap_owners
    if not owners:
        # Defensive fallback: unattributable edits are held back entirely.
        return KIND_FILE, tuple(range(statement_count))
    return kind, tuple(sorted(owners))


def _tight_source_slice(
    templated_file: TemplatedFile, templated_start: int, templated_stop: int
) -> slice:
    """Map a templated region to a tight source range for display.

    Literal slices map 1:1, so only the source text corresponding to the
    region interior is covered. Non-literal slices (template tags) which
    intersect the interior are included wholesale. Zero-length slices on the
    region boundaries are excluded so leading/trailing template tags do not
    widen the displayed statement text.
    """
    low: Optional[int] = None
    high: Optional[int] = None
    for file_slice in templated_file.sliced_file:
        t_start, t_stop = (
            file_slice.templated_slice.start,
            file_slice.templated_slice.stop,
        )
        # Positive-length intersection with the region interior.
        intersect_start = max(t_start, templated_start)
        intersect_stop = min(t_stop, templated_stop)
        if intersect_start >= intersect_stop:
            continue
        s_start, s_stop = (
            file_slice.source_slice.start,
            file_slice.source_slice.stop,
        )
        if file_slice.slice_type == "literal" and s_stop - s_start == t_stop - t_start:
            mapped_start = s_start + (intersect_start - t_start)
            mapped_stop = s_start + (intersect_stop - t_start)
        elif s_start < s_stop:
            # Non-literal content within the region: keep the source bounds.
            mapped_start, mapped_stop = s_start, s_stop
        else:
            continue
        low = mapped_start if low is None else min(low, mapped_start)
        high = mapped_stop if high is None else max(high, mapped_stop)
    if low is None or high is None:
        return templated_file.templated_slice_to_source_slice(
            slice(templated_start, templated_stop)
        )
    return slice(low, high)


@dataclass(frozen=True)
class _Hunk:
    """An internal diff hunk with attribution."""

    source_start: int
    source_stop: int
    original: str
    fixed: str
    kind: str
    owners: tuple[int, ...]
    rule_codes: tuple[str, ...]

    def as_edit(self, reason: str) -> StatementEdit:
        return StatementEdit(
            kind=self.kind,
            statement_indices=self.owners,
            source_slice=slice(self.source_start, self.source_stop),
            original_source=self.original,
            fixed_source=self.fixed,
            rule_codes=self.rule_codes,
            reason=reason,
        )


def _hunk_rule_codes(
    linted_file: "LintedFile",
    source_start: int,
    source_stop: int,
) -> tuple[str, ...]:
    """Best effort association of fixable rule codes with a hunk."""
    codes: list[str] = []
    seen: set[str] = set()
    for violation in linted_file.get_violations(fixable=True):
        if not isinstance(violation, SQLLintError) or not violation.segment.pos_marker:
            continue
        violation_slice = violation.segment.pos_marker.source_slice
        if source_start == source_stop:
            overlaps = violation_slice.start <= source_start <= violation_slice.stop
        elif violation_slice.start == violation_slice.stop:
            overlaps = source_start <= violation_slice.start <= source_stop
        else:
            overlaps = (
                source_start < violation_slice.stop
                and source_stop > violation_slice.start
            )
        if overlaps:
            code = violation.rule_code()
            if code not in seen:
                seen.add(code)
                codes.append(code)
    return tuple(codes)


def _edit_reason(kind: str, owners: tuple[int, ...]) -> str:
    """Human-readable explanation for an ownership decision."""
    displayed = tuple(index + 1 for index in owners)
    if kind == KIND_STATEMENT:
        return f"Edit within statement {displayed[0]}."
    if kind == KIND_BOUNDARY:
        return (
            f"Edit on the boundary between statements {displayed[0]} and "
            f"{displayed[1]}; it is only applied when both statements are "
            "selected."
        )
    if kind == KIND_TEMPLATE:
        return (
            "Edit in a shared template fragment rendered into statements "
            f"{displayed}; it is only applied when all of those statements "
            "are selected."
        )
    if kind == KIND_BOUNDARY_TEMPLATE:
        return (
            "Edit both on a statement boundary and in a shared template "
            f"fragment (statements {displayed}); it is only applied when all "
            "of those statements are selected."
        )
    return (
        "Edit outside any statement (leading/trailing file padding); it is "
        "only applied when all statements are selected."
    )


def build_statement_fix_preview(linted_file: "LintedFile") -> StatementFixPreview:
    """Build a per-statement fix preview for a linted file.

    The linted file must have been produced with fixing enabled (``fix=True``)
    for the preview to contain edits.

    Args:
        linted_file: The result of a fixing lint run.

    Returns:
        A :class:`StatementFixPreview` attributing every available edit to
        source statements.

    Raises:
        ValueError: If the file did not template or parse successfully, since
            statement attribution is not possible without either.
    """
    templated_file = linted_file.templated_file
    tree = linted_file.tree
    if templated_file is None or tree is None:
        raise ValueError(
            "Cannot build a statement fix preview without a successfully "
            "templated and parsed file."
        )

    source_str = _normalise_newlines(templated_file.source_str)
    fixed_str, changed = linted_file.fix_string()

    regions, statement_count = _build_regions(tree, templated_file)
    linter_logger.debug(
        "Statement fix preview for %s: %s statements templated",
        linted_file.path,
        statement_count,
    )

    raw_hunks = _diff_hunks(source_str, fixed_str) if changed else []
    hunks: list[_Hunk] = []
    for (
        source_start,
        source_stop,
        _fixed_start,
        _fixed_stop,
        original,
        fixed,
    ) in raw_hunks:
        kind, owners = _attribute_hunk(
            source_start,
            source_stop,
            templated_file,
            regions,
            statement_count,
        )
        hunks.append(
            _Hunk(
                source_start=source_start,
                source_stop=source_stop,
                original=original,
                fixed=fixed,
                kind=kind,
                owners=owners,
                rule_codes=_hunk_rule_codes(linted_file, source_start, source_stop),
            )
        )

    edits = [hunk.as_edit(_edit_reason(hunk.kind, hunk.owners)) for hunk in hunks]

    groups: list[StatementFixGroup] = []
    for index, statement in enumerate(
        seg for seg in tree.get_children("statement", "unparsable") if seg.pos_marker
    ):
        # Recover the region (including any extended delimiter range).
        region = next(
            region
            for region in regions
            if region.kind == "statement" and region.owners == (index,)
        )
        source_slice = _tight_source_slice(templated_file, region.start, region.stop)
        owned_hunks = [
            hunk
            for hunk in hunks
            if hunk.kind == KIND_STATEMENT and hunk.owners == (index,)
        ]
        original_source = source_str[source_slice]
        fixed_source = _render_range(source_str, hunks, source_slice, {index})
        line_no, line_pos = templated_file.get_line_pos_of_char_pos(
            source_slice.start, source=True
        )
        rule_codes: list[str] = []
        for hunk in owned_hunks:
            for code in hunk.rule_codes:
                if code not in rule_codes:
                    rule_codes.append(code)
        groups.append(
            StatementFixGroup(
                index=index,
                source_slice=source_slice,
                line_no=line_no,
                line_pos=line_pos,
                original_source=original_source,
                fixed_source=fixed_source,
                rule_codes=tuple(rule_codes),
                edits=tuple(
                    hunk.as_edit(_edit_reason(hunk.kind, hunk.owners))
                    for hunk in owned_hunks
                ),
            )
        )

    return StatementFixPreview(
        path=linted_file.path,
        encoding=linted_file.encoding,
        source_hash=hashlib.sha256(source_str.encode("utf-8")).hexdigest(),
        statements=groups,
        edits=edits,
        _source_str=source_str,
        _fixed_str=fixed_str,
        _hunks=hunks,
    )


# -----------------------------------------------------------------------------
# Rendering selections
# -----------------------------------------------------------------------------


def _hunk_allowed(hunk: _Hunk, selected: set[int]) -> bool:
    return bool(hunk.owners) and all(owner in selected for owner in hunk.owners)


def _render_range(
    source_str: str,
    hunks: Sequence[_Hunk],
    source_slice: slice,
    selected: set[int],
) -> str:
    """Render a source range applying only hunks allowed by ``selected``.

    Hunks shared with statements outside the selection are left as the
    original text.
    """
    low, high = source_slice.start, source_slice.stop
    pieces: list[str] = []
    cursor = low
    for hunk in hunks:
        if hunk.source_stop <= low or hunk.source_start >= high:
            continue
        start = max(hunk.source_start, low)
        stop = min(hunk.source_stop, high)
        pieces.append(source_str[cursor:start])
        if _hunk_allowed(hunk, selected):
            # Hunks are only ever partially clipped at the range boundary for
            # display. Within a statement region this should not occur
            # because shared boundary hunks are disjoint from it.
            if start == hunk.source_start and stop == hunk.source_stop:
                pieces.append(hunk.fixed)
            else:
                # Fall back to the original for cross-region fragments.
                pieces.append(source_str[start:stop])
        else:
            pieces.append(source_str[start:stop])
        cursor = stop
    pieces.append(source_str[cursor:high])
    return "".join(pieces)


def render_fixed_source(
    preview: StatementFixPreview, selected_statements: Sequence[int]
) -> str:
    """Render the fixed source for a selection of statement indices.

    Unselected statements and edits shared with unselected statements are
    left byte-for-byte as the original source.

    Args:
        preview: The preview to render from.
        selected_statements: Zero-based statement indices to accept.

    Returns:
        The rendered source string.

    Raises:
        ValueError: If a selected index does not exist in the preview.
    """
    rendered, _, _ = _render_selection(preview, selected_statements)
    return rendered


def _render_selection(
    preview: StatementFixPreview, selected_statements: Sequence[int]
) -> tuple[str, list[_Hunk], list[_Hunk]]:
    statement_count = len(preview.statements)
    selected = set(selected_statements)
    invalid = sorted(index for index in selected if not 0 <= index < statement_count)
    if invalid:
        raise ValueError(
            f"Unknown statement indices for {preview.path!r}: {invalid}. "
            f"Valid indices are 0 to {statement_count - 1}."
        )

    pieces: list[str] = []
    cursor = 0
    applied: list[_Hunk] = []
    held: list[_Hunk] = []
    for hunk in preview._hunks:
        pieces.append(preview._source_str[cursor : hunk.source_start])
        if _hunk_allowed(hunk, selected):
            pieces.append(hunk.fixed)
            applied.append(hunk)
        else:
            pieces.append(hunk.original)
            held.append(hunk)
        cursor = hunk.source_stop
    pieces.append(preview._source_str[cursor:])
    return "".join(pieces), applied, held


def _read_current_source(
    preview: StatementFixPreview, current_source: Optional[str]
) -> str:
    """Resolve and normalise the source currently available to apply to."""
    if current_source is not None:
        return _normalise_newlines(current_source)
    if preview.path and os.path.exists(preview.path):
        with open(
            preview.path, encoding=preview.encoding, errors="backslashreplace"
        ) as handle:
            return _normalise_newlines(handle.read())
    # String input without a backing file: the preview source is all we have.
    return preview._source_str


def apply_statement_fixes(
    preview: StatementFixPreview,
    selected_statements: Sequence[int],
    *,
    current_source: Optional[str] = None,
    parse_checker: Optional[Callable[[str], Sequence[SQLBaseError]]] = None,
    persist: bool = False,
    linted_file: Optional["LintedFile"] = None,
) -> StatementFixResult:
    """Apply the edits of a selection of statements.

    Args:
        preview: The preview describing available edits.
        selected_statements: Zero-based statement indices to accept.
        current_source: The current source text to apply against. When
            omitted, the preview's backing file is re-read from disk (if it
            exists), otherwise the previewed source is used. If this does not
            match the previewed source exactly, application is refused.
        parse_checker: Optional callable which reparses the rendered source
            and returns parse/template/lex violations. A non-empty result
            blocks application (and persistence). Callers using
            :meth:`sqlfluff.core.linter.linter.Linter.apply_statement_fixes`
            get this automatically.
        persist: Whether to write the result back to the preview path.
        linted_file: Required when ``persist`` is True, to reuse its safe
            file-writing routine.

    Returns:
        A :class:`StatementFixResult`.

    Raises:
        StaleSourceError: If the current source differs from the previewed
            source.
        ValueError: For invalid statement indices or persistence without a
            backing file.
    """
    candidate_source = _read_current_source(preview, current_source)
    candidate_hash = hashlib.sha256(candidate_source.encode("utf-8")).hexdigest()
    if candidate_hash != preview.source_hash:
        linter_logger.warning(
            "Refusing stale statement fix application for %s (source hash "
            "changed since the preview was built).",
            preview.path,
        )
        raise StaleSourceError(preview.path, preview.source_hash, candidate_hash)

    rendered, applied_hunks, held_hunks = _render_selection(
        preview, selected_statements
    )

    violations: tuple[SQLBaseError, ...] = ()
    if parse_checker is not None:
        reparse_violations = [
            violation
            for violation in parse_checker(rendered)
            if isinstance(violation, _REPARSE_FAILURE_TYPES)
        ]
        violations = tuple(reparse_violations)

    success = not violations
    written_path: Optional[str] = None
    if success and persist:
        if linted_file is None:
            raise ValueError(
                "Persisting statement fixes requires the originating "
                "LintedFile; pass linted_file=... or use Linter."
                "apply_statement_fixes()."
            )
        linted_file._safe_create_replace_file(
            preview.path, preview.path, rendered, preview.encoding
        )
        written_path = preview.path

    return StatementFixResult(
        success=success,
        fixed_source=rendered,
        applied_edits=tuple(
            hunk.as_edit(_edit_reason(hunk.kind, hunk.owners)) for hunk in applied_hunks
        ),
        held_edits=tuple(
            hunk.as_edit(_edit_reason(hunk.kind, hunk.owners)) for hunk in held_hunks
        ),
        violations=violations,
        path=written_path,
    )

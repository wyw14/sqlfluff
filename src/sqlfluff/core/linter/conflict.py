"""Preview and resolve conflicts between source-space fix patches.

When templated code is linted (for example Jinja loops or conditional
config variants) a single fragment of the *source* file may produce several
sections of rendered SQL. Fixes for those sections are mapped back onto the
source file as :class:`~sqlfluff.core.linter.patch.FixPatch` objects, so
several suggestions may land on the same source range even though they came
from different places in the rendered SQL.

Conflict analysis here is therefore performed purely in **source space**
(``source_slice`` / ``fixed_raw``), never in templated space. The module
never guesses which suggestion "wins": :func:`build_fix_preview` groups the
suggestions by source range and classifies the relationship between the
distinct edits as:

* :py:attr:`EditRelation.IDENTICAL` - the same source range and the same
  replacement (e.g. the same fix found on every iteration of a loop, or in
  more than one render variant). Applied once.
* :py:attr:`EditRelation.COMPATIBLE` - disjoint source ranges which can be
  applied together.
* :py:attr:`EditRelation.EXCLUSIVE` - overlapping source ranges with
  different replacements. A user must choose; no edit is silently preferred.

:func:`apply_source_patches` (and ``LintedFile.apply_fix_selection``) then
re-check the *current* source text against the text recorded when each
patch was generated and refuse to apply stale positions.
"""

import enum
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from sqlfluff.core.linter.patch import FixPatch

linter_logger = logging.getLogger("sqlfluff.linter")


class EditRelation(enum.Enum):
    """How two (or more) suggested source edits relate to each other."""

    IDENTICAL = "identical"
    COMPATIBLE = "compatible"
    EXCLUSIVE = "exclusive"


class ConflictingEditsError(ValueError):
    """Raised when a selection contains edits which cannot be applied together.

    This represents an ambiguous choice: the caller must resolve the
    :py:attr:`EditRelation.EXCLUSIVE` alternatives rather than asking for
    both to be applied.
    """


class StaleSourceError(ValueError):
    """Raised when a patch no longer matches the current source text.

    The source file has changed since the patches were generated, so the
    recorded positions are stale. The file must be re-linted before any of
    these edits can safely be applied.
    """


@dataclass(frozen=True)
class PatchSuggestion:
    """A single patch suggested for a source range, with provenance.

    Attributes:
        patch: The suggested source patch.
        origin: Where the suggestion came from (e.g. ``"root_variant"`` or
            ``"alternate_variant_1"``). An identical edit suggested by more
            than one origin appears as more than one ``PatchSuggestion`` on
            the same :class:`SourceEditPreview`.
    """

    patch: FixPatch
    origin: str


@dataclass
class SourceEditPreview:
    """A distinct edit on one source range, with all of its suggestions.

    Attributes:
        source_slice: The range in the source file which the edit affects.
        fixed_raw: The text which would replace the content of that range.
        source_str: The content expected to be present at the range when the
            patch was generated. Used to detect stale positions.
        suggestions: Every suggestion which converges on this exact edit
            (same range and replacement). More than one entry means the edit
            was suggested identically from more than one origin.
        alternatives: Other distinct edits whose source ranges overlap this
            one and which make different replacements. These are mutually
            exclusive with this edit.
    """

    source_slice: slice
    fixed_raw: str
    source_str: str
    suggestions: list[PatchSuggestion] = field(default_factory=list)
    alternatives: list["SourceEditPreview"] = field(default_factory=list)

    @classmethod
    def from_patch(cls, patch: FixPatch) -> "SourceEditPreview":
        """Build an initial preview for a single suggested patch."""
        return cls(
            source_slice=patch.source_slice,
            fixed_raw=patch.fixed_raw,
            source_str=patch.source_str,
        )

    def add_suggestion(self, patch: FixPatch, origin: str) -> None:
        """Record another identical suggestion of this edit."""
        self.suggestions.append(PatchSuggestion(patch=patch, origin=origin))

    @property
    def relation(self) -> EditRelation:
        """The relation this edit has with the other suggested edits."""
        if self.alternatives:
            return EditRelation.EXCLUSIVE
        if len(self.suggestions) > 1:
            return EditRelation.IDENTICAL
        return EditRelation.COMPATIBLE

    @property
    def patch(self) -> FixPatch:
        """A representative patch for applying this edit."""
        return self.suggestions[0].patch

    @property
    def origins(self) -> list[str]:
        """The origins (in suggestion order) which suggested this edit."""
        return [suggestion.origin for suggestion in self.suggestions]


@dataclass
class FixConflictPreview:
    """A preview of all distinct source-space edits for one source file.

    Edits are ordered by their source range. Edits suggested identically by
    more than one origin are collapsed into one entry; mutually exclusive
    edits cross-reference each other via
    :attr:`SourceEditPreview.alternatives`.
    """

    edits: list[SourceEditPreview]

    def __iter__(self) -> Iterable[SourceEditPreview]:
        return iter(self.edits)

    def __len__(self) -> int:
        return len(self.edits)

    @property
    def has_conflicts(self) -> bool:
        """Return whether any mutually exclusive edits are present."""
        return any(edit.relation is EditRelation.EXCLUSIVE for edit in self.edits)

    def get_edits(self, relation: EditRelation) -> list[SourceEditPreview]:
        """Return the previewed edits with the given relation."""
        return [edit for edit in self.edits if edit.relation is relation]


def classify_source_edits(first: FixPatch, second: FixPatch) -> EditRelation:
    """Classify the relationship between two patches in source space.

    Only source positions and replacements are consulted; the templated
    positions of the patches are deliberately ignored (one source fragment
    can render in many places, and renders can differ between variants).

    Args:
        first: The first source patch.
        second: The second source patch.

    Returns:
        :py:attr:`EditRelation.IDENTICAL` when both patches replace the same
        source range with the same text,
        :py:attr:`EditRelation.EXCLUSIVE` when the ranges overlap (or are the
        same insertion point) but the replacements differ, and
        :py:attr:`EditRelation.COMPATIBLE` when the ranges are disjoint and
        the edits can be applied together.
    """
    # Identical edits compare on source range *and* replacement. Using the
    # dedupe tuple also treats point insertions with unequal-but-equivalent
    # slice objects (e.g. ``slice(2, 2)`` and ``slice(2, 2, 1)``) correctly.
    if first.dedupe_tuple() == second.dedupe_tuple():
        return EditRelation.IDENTICAL

    first_start = int(first.source_slice.start)
    first_stop = int(first.source_slice.stop)
    second_start = int(second.source_slice.start)
    second_stop = int(second.source_slice.stop)

    # Distinct insertions offered at the very same source point are
    # ambiguous and therefore mutually exclusive.
    if first_start == first_stop and second_start == second_stop:
        if first_start == second_start:
            return EditRelation.EXCLUSIVE
    # The same range with a different replacement is always exclusive.
    elif first_start == second_start and first_stop == second_stop:
        return EditRelation.EXCLUSIVE

    # Otherwise compare proper ranges: strict overlap means exclusive,
    # merely touching (or fully disjoint) ranges remain compatible.
    if max(first_start, second_start) < min(first_stop, second_stop):
        return EditRelation.EXCLUSIVE
    return EditRelation.COMPATIBLE


def build_fix_preview(
    patch_buffers: Sequence[Sequence[FixPatch]],
    origins: Sequence[str] | None = None,
) -> FixConflictPreview:
    """Build a source-space conflict preview from per-variant patch lists.

    Args:
        patch_buffers: One list of :class:`FixPatch` objects per render
            variant (with the root variant conventionally first). The same
            source edit appearing in more than one buffer is recognised as
            :py:attr:`EditRelation.IDENTICAL` rather than applied twice.
        origins: Optional labels for the buffers (used for provenance). If
            not supplied, buffers are labelled ``"variant 0"`` and so on.

    Returns:
        A :class:`FixConflictPreview` ordered by source range.
    """
    if origins is None:
        origins = [f"variant {idx}" for idx in range(len(patch_buffers))]
    elif len(origins) != len(patch_buffers):
        raise ValueError(
            "origins must contain one label per patch buffer: "
            f"got {len(origins)} labels for {len(patch_buffers)} buffers."
        )

    # Collapse identical suggestions (same source range + replacement),
    # remembering every origin which suggested them.
    previews: dict[tuple[tuple[int, int], str], SourceEditPreview] = {}
    for patch_buffer, origin in zip(patch_buffers, origins):
        for patch in patch_buffer:
            key = patch.dedupe_tuple()
            existing = previews.get(key)
            if existing is None:
                preview = SourceEditPreview.from_patch(patch)
                preview.add_suggestion(patch, origin)
                previews[key] = preview
            else:
                existing.add_suggestion(patch, origin)

    edits = sorted(
        previews.values(),
        key=lambda edit: (
            int(edit.source_slice.start),
            int(edit.source_slice.stop),
        ),
    )

    # Cross-link mutually exclusive alternatives. Every entry keeps the
    # full list of competing edits: nothing is picked or dropped here.
    for idx, edit in enumerate(edits):
        for other in edits[idx + 1 :]:
            relation = classify_source_edits(edit.patch, other.patch)
            if relation is EditRelation.EXCLUSIVE:
                edit.alternatives.append(other)
                other.alternatives.append(edit)

    preview = FixConflictPreview(edits=edits)
    if preview.has_conflicts:
        linter_logger.info(
            "Fix conflict preview found %s mutually exclusive edit(s).",
            len(preview.get_edits(EditRelation.EXCLUSIVE)),
        )
    return preview


def deduplicate_patches(patches: Sequence[FixPatch]) -> list[FixPatch]:
    """Remove identical patches (same source range and replacement)."""
    unique: list[FixPatch] = []
    seen: set[tuple[tuple[int, int], str]] = set()
    for patch in patches:
        key = patch.dedupe_tuple()
        if key in seen:
            continue
        seen.add(key)
        unique.append(patch)
    return unique


def validate_patch_selection(patches: Sequence[FixPatch]) -> list[FixPatch]:
    """Check a user selection can be applied as one set of edits.

    Identical patches are collapsed. Mutually exclusive patches raise
    :class:`ConflictingEditsError` so that the ambiguity is resolved by the
    caller instead of one edit silently winning.

    Args:
        patches: The patches selected for application.

    Returns:
        The deduplicated selection in input order.
    """
    unique_patches = deduplicate_patches(patches)
    for idx, first in enumerate(unique_patches):
        for second in unique_patches[idx + 1 :]:
            relation = classify_source_edits(first, second)
            if relation is EditRelation.EXCLUSIVE:
                raise ConflictingEditsError(
                    "Cannot apply mutually exclusive edits together: "
                    f"{first.dedupe_tuple()!r} vs {second.dedupe_tuple()!r}. "
                    "Choose between the alternatives shown in the fix "
                    "conflict preview."
                )
    return unique_patches


def verify_source_patches(source_str: str, patches: Iterable[FixPatch]) -> None:
    """Re-check patches against the current source text.

    Args:
        source_str: The current source file content.
        patches: The patches intended for application.

    Raises:
        StaleSourceError: If a patch's range falls outside the current
            source or the source text at that range no longer matches the
            content recorded when the patch was generated.
    """
    source_len = len(source_str)
    for patch in patches:
        start = int(patch.source_slice.start)
        stop = int(patch.source_slice.stop)
        if not (0 <= start <= stop <= source_len):
            raise StaleSourceError(
                f"Stale fix position: source range slice({start}, {stop}) "
                f"does not fit in the current source (length {source_len}). "
                "Re-lint the file to regenerate the fixes."
            )
        current_text = source_str[start:stop]
        if current_text != patch.source_str:
            raise StaleSourceError(
                f"Stale fix position: source text at slice({start}, {stop}) "
                "has changed since the fixes were generated. Expected "
                f"{patch.source_str!r}, found {current_text!r}. Re-lint the "
                "file to regenerate the fixes."
            )


def apply_source_patches(source_str: str, patches: Sequence[FixPatch]) -> str:
    """Apply a selected set of source patches after re-verifying the source.

    Identical patches are applied once, compatible patches are applied
    together and mutually exclusive patches raise
    :class:`ConflictingEditsError`. Every patch is re-checked against the
    current source text first; if the file has moved on,
    :class:`StaleSourceError` is raised and no edit is made.

    Args:
        source_str: The current source file content.
        patches: The selected patches to apply.

    Returns:
        The source content with the selected edits applied.
    """
    unique_patches = validate_patch_selection(patches)
    verify_source_patches(source_str, unique_patches)

    # Work backwards through the (disjoint) ranges so earlier offsets remain
    # valid while later edits are applied.
    fixed_source = source_str
    for patch in sorted(
        unique_patches,
        key=lambda candidate: (
            int(candidate.source_slice.start),
            int(candidate.source_slice.stop),
        ),
        reverse=True,
    ):
        start = int(patch.source_slice.start)
        stop = int(patch.source_slice.stop)
        fixed_source = fixed_source[:start] + patch.fixed_raw + fixed_source[stop:]
    return fixed_source

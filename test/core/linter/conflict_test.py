"""Tests for source-space fix conflict preview and verified application.

These cover :mod:`sqlfluff.core.linter.conflict`, which classifies fix
suggestions mapped back onto a *source* file (not the rendered SQL) and
re-checks the current source text before applying any user selection.
"""

import pytest

from sqlfluff.core import Linter
from sqlfluff.core.config import FluffConfig
from sqlfluff.core.linter import (
    ConflictingEditsError,
    EditRelation,
    StaleSourceError,
    apply_source_patches,
    build_fix_preview,
)
from sqlfluff.core.linter.conflict import (
    classify_source_edits,
    deduplicate_patches,
    validate_patch_selection,
    verify_source_patches,
)
from sqlfluff.core.linter.patch import FixPatch, _patches_conflict


def make_patch(
    source_start: int,
    source_stop: int,
    fixed_raw: str,
    source_str: str = "",
    *,
    templated_start: int | None = None,
    templated_stop: int | None = None,
    patch_category: str = "literal",
    step: int | None = None,
) -> FixPatch:
    """Construct a FixPatch for tests, defaulting templated positions to source."""
    t_start = source_start if templated_start is None else templated_start
    t_stop = source_stop if templated_stop is None else templated_stop
    source_slice = (
        slice(source_start, source_stop, step)
        if step
        else slice(source_start, source_stop)
    )
    return FixPatch(
        slice(t_start, t_stop),
        fixed_raw,
        patch_category,
        source_slice,
        "",
        source_str,
    )


# --------------------------------------------------------------------------
# Pairwise classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "first,second,expected",
    [
        # The same source range and replacement is identical...
        (
            make_patch(2, 3, "X", "b"),
            make_patch(2, 3, "X", "b"),
            EditRelation.IDENTICAL,
        ),
        # ...even when the suggestions came from different places in the
        # rendered SQL. Templated positions must not drive the decision.
        (
            make_patch(2, 3, "X", "b", templated_start=0, templated_stop=1),
            make_patch(2, 3, "X", "b", templated_start=9, templated_stop=10),
            EditRelation.IDENTICAL,
        ),
        # Point insertions represented by unequal-but-equivalent slice
        # objects (e.g. with an explicit step) must still compare identical.
        (
            make_patch(4, 4, "A", "", patch_category="mid_point"),
            make_patch(4, 4, "A", "", patch_category="mid_point", step=1),
            EditRelation.IDENTICAL,
        ),
        # The same range with a different replacement is mutually exclusive,
        # again regardless of (even identical) templated positions.
        (
            make_patch(2, 3, "X", "b", templated_start=7, templated_stop=8),
            make_patch(2, 3, "Y", "b", templated_start=7, templated_stop=8),
            EditRelation.EXCLUSIVE,
        ),
        # Strictly overlapping ranges are exclusive.
        (
            make_patch(2, 6, "X", "bcde"),
            make_patch(5, 8, "Y", "efg"),
            EditRelation.EXCLUSIVE,
        ),
        # One range enclosing another is exclusive.
        (
            make_patch(2, 8, "X", "bcdefg"),
            make_patch(4, 6, "Y", "de"),
            EditRelation.EXCLUSIVE,
        ),
        # Different insertions offered at the same source point are
        # exclusive (it would be ambiguous which came first).
        (
            make_patch(4, 4, "A", "", patch_category="mid_point"),
            make_patch(4, 4, "B", "", patch_category="mid_point", step=1),
            EditRelation.EXCLUSIVE,
        ),
        # Disjoint ranges are compatible.
        (
            make_patch(2, 3, "X", "b"),
            make_patch(5, 6, "Y", "f"),
            EditRelation.COMPATIBLE,
        ),
        # Merely touching ranges share a boundary but no characters, so they
        # are jointly applicable.
        (
            make_patch(2, 5, "X", "bcd"),
            make_patch(5, 8, "Y", "efg"),
            EditRelation.COMPATIBLE,
        ),
        # Insertions at different points are compatible.
        (
            make_patch(2, 2, "A", "", patch_category="mid_point"),
            make_patch(5, 5, "B", "", patch_category="mid_point"),
            EditRelation.COMPATIBLE,
        ),
    ],
)
def test__conflict__classify_source_edits(first, second, expected):
    """Pairwise classification must be source-space based and symmetric."""
    assert classify_source_edits(first, second) is expected
    assert classify_source_edits(second, first) is expected
    # The legacy boolean predicate must agree about exclusivity.
    assert _patches_conflict(first, second) is (expected is EditRelation.EXCLUSIVE)


def test__conflict__identical_point_insertions_do_not_conflict():
    """Regression: equivalent zero slices with the same text aren't conflicts."""
    first = make_patch(2, 2, "A", "", patch_category="mid_point")
    second = make_patch(2, 2, "A", "", patch_category="mid_point", step=1)
    assert not _patches_conflict(first, second)


# --------------------------------------------------------------------------
# Preview grouping
# --------------------------------------------------------------------------


def test__conflict__build_fix_preview_empty():
    """No variants means an empty preview."""
    preview = build_fix_preview([])
    assert list(preview) == []
    assert len(preview) == 0
    assert not preview.has_conflicts


def test__conflict__preview_collapses_identical_edits_with_origins():
    """The same edit from two variants is one IDENTICAL preview entry."""
    root_patch = make_patch(2, 3, "X", "b", templated_start=0, templated_stop=1)
    alt_patch = make_patch(2, 3, "X", "b", templated_start=9, templated_stop=10)

    preview = build_fix_preview(
        [[root_patch], [alt_patch]],
        origins=("root_variant", "alternate_variant_1"),
    )

    assert len(preview) == 1
    edit = preview.edits[0]
    assert edit.source_slice == slice(2, 3)
    assert edit.fixed_raw == "X"
    assert edit.relation is EditRelation.IDENTICAL
    assert edit.origins == ["root_variant", "alternate_variant_1"]
    assert [suggestion.patch for suggestion in edit.suggestions] == [
        root_patch,
        alt_patch,
    ]
    assert edit.patch == root_patch
    assert not edit.alternatives


def test__conflict__preview_compatible_edits_get_individual_entries():
    """Non-conflicting edits must still each form their own preview entry."""
    first = make_patch(1, 2, "A", "b")
    second = make_patch(5, 6, "B", "f")

    preview = build_fix_preview([[first, second]])

    assert len(preview) == 2
    assert [edit.source_slice for edit in preview] == [slice(1, 2), slice(5, 6)]
    assert {edit.relation for edit in preview} == {EditRelation.COMPATIBLE}
    assert all(edit.origins == ["variant 0"] for edit in preview)
    assert not preview.has_conflicts


def test__conflict__preview_links_exclusive_alternatives_without_picking():
    """Competing edits cross-reference each other; nothing is preferred."""
    first = make_patch(2, 3, "X", "b")
    second = make_patch(2, 3, "Y", "b")
    disjoint = make_patch(8, 9, "Z", "i")

    preview = build_fix_preview(
        [[first, disjoint], [second]],
        origins=("root_variant", "alternate_variant_1"),
    )

    assert len(preview) == 3
    exclusive = preview.get_edits(EditRelation.EXCLUSIVE)
    compatible = preview.get_edits(EditRelation.COMPATIBLE)
    identical = preview.get_edits(EditRelation.IDENTICAL)
    assert preview.has_conflicts
    assert len(exclusive) == 2
    assert len(compatible) == 1
    assert identical == []
    x_preview, y_preview = (
        (exclusive[0], exclusive[1])
        if exclusive[0].fixed_raw == "X"
        else (exclusive[1], exclusive[0])
    )
    assert x_preview.origins == ["root_variant"]
    assert y_preview.origins == ["alternate_variant_1"]
    # Both alternatives are linked both ways - neither was dropped/chosen.
    assert x_preview.alternatives == [y_preview]
    assert y_preview.alternatives == [x_preview]
    assert compatible[0].fixed_raw == "Z"
    assert compatible[0].alternatives == []


def test__conflict__preview_deduplicates_within_one_buffer():
    """Duplicate suggestions in a single buffer are still collapsed."""
    patch = make_patch(2, 3, "X", "b")
    preview = build_fix_preview([[patch, patch]], origins=("root_variant",))
    assert len(preview) == 1
    assert preview.edits[0].relation is EditRelation.IDENTICAL


def test__conflict__preview_default_origins():
    """Default origin labels identify buffers by index."""
    preview = build_fix_preview(
        [[make_patch(1, 2, "A", "b")], [make_patch(5, 6, "B", "f")]]
    )
    assert [edit.origins[0] for edit in preview] == ["variant 0", "variant 1"]


def test__conflict__preview_origins_length_mismatch_raises():
    """There must be one origin label per patch buffer."""
    with pytest.raises(ValueError, match="one label per patch buffer"):
        build_fix_preview([[], []], origins=("only_one",))


# --------------------------------------------------------------------------
# Selection validation and application
# --------------------------------------------------------------------------


def test__conflict__deduplicate_and_validate_patches():
    """Identical patches collapse; an ordered selection is returned."""
    first = make_patch(2, 3, "X", "b")
    duplicate = make_patch(2, 3, "X", "b", templated_start=8, templated_stop=9)
    second = make_patch(5, 6, "Y", "f")

    assert deduplicate_patches([first, duplicate, second]) == [first, second]
    assert validate_patch_selection([duplicate, first, second]) == [
        duplicate,
        second,
    ]


def test__conflict__validate_selection_rejects_exclusive_edits():
    """A selection mixing mutually exclusive edits must be refused."""
    with pytest.raises(ConflictingEditsError, match="mutually exclusive"):
        validate_patch_selection(
            [make_patch(2, 6, "X", "bcde"), make_patch(5, 8, "Y", "efg")]
        )


def test__conflict__apply_source_patches_replaces_inserts_and_deletes():
    """Disjoint replacement, insertion and deletion edits apply together."""
    source = "abcdef"
    patches = [
        make_patch(1, 2, "B", "b"),  # replace
        make_patch(3, 3, "XY", "", patch_category="mid_point"),  # insert
        make_patch(5, 6, "", "f"),  # delete
    ]
    assert apply_source_patches(source, patches) == "aBcXYde"


def test__conflict__apply_applies_identical_edit_once():
    """Identical patches from different variants apply a single edit."""
    first = make_patch(2, 3, "X", "c", templated_start=0, templated_stop=1)
    second = make_patch(2, 3, "X", "c", templated_start=9, templated_stop=10)
    assert apply_source_patches("abcdef", [first, second]) == "abXdef"


def test__conflict__apply_rejects_exclusive_selection():
    """Applying mutually exclusive edits raises rather than overwriting."""
    with pytest.raises(ConflictingEditsError):
        apply_source_patches(
            "abcdef",
            [make_patch(2, 3, "X", "c"), make_patch(2, 3, "Y", "c")],
        )


def test__conflict__apply_rejects_stale_source_text():
    """Changed source text at the recorded range rejects the stale patch."""
    patch = make_patch(2, 3, "X", "c")
    with pytest.raises(StaleSourceError, match="has changed"):
        apply_source_patches("abXdef", [patch])


def test__conflict__apply_rejects_out_of_range_positions():
    """Positions past the end (or reversed) are treated as stale."""
    source = "ab"
    with pytest.raises(StaleSourceError, match="does not fit"):
        verify_source_patches(source, [make_patch(1, 3, "X", "bc")])
    with pytest.raises(StaleSourceError, match="does not fit"):
        verify_source_patches(source, [make_patch(2, 1, "X", "")])


def test__conflict__verify_source_patches_accepts_matching_text():
    """Verification is silent when the recorded source text still matches."""
    patch = make_patch(2, 3, "X", "c")
    verify_source_patches("abcdef", [patch])


# --------------------------------------------------------------------------
# LintedFile integration
# --------------------------------------------------------------------------


def test__linted_file__fix_conflict_preview_and_verified_selection():
    """A real fixed file carries a preview; selection applies with checks."""
    sql = "SELECT\n  1\n"
    config = FluffConfig(overrides={"dialect": "ansi", "rules": "LT02"})
    linted = Linter(config=config).lint_string(sql, fix=True)

    preview = linted.fix_conflict_preview()
    assert preview is linted.fix_preview
    assert len(preview) >= 1
    assert all(edit.origins == ["root_variant"] for edit in preview)
    # LT02 produces a single compatible edit in this example.
    assert all(edit.relation is EditRelation.COMPATIBLE for edit in preview)

    # Applying the whole (non-conflicting) preview matches fix_string().
    selected = [edit.patch for edit in preview]
    assert linted.apply_fix_selection(selected) == "SELECT\n    1\n"
    fixed_string, changed = linted.fix_string()
    assert changed
    assert linted.apply_fix_selection(selected) == fixed_string

    # If the source moved on after linting, stale positions are rejected.
    with pytest.raises(StaleSourceError):
        linted.apply_fix_selection(selected, current_source_str="SELECT\n 1\n")
    # Identical duplicates in a selection collapse harmlessly.
    assert linted.apply_fix_selection(selected + selected) == fixed_string


def test__linted_file__fix_preview_fallback_without_precomputed_patches():
    """Preview is generated from patches or the tree when not precomputed."""
    sql = "SELECT\n  1\n"
    config = FluffConfig(overrides={"dialect": "ansi", "rules": "LT02"})
    linter = Linter(config=config)

    # Lint-only runs carry no preview and no patches: preview is empty.
    linted_only = linter.lint_string(sql, fix=False)
    assert linted_only.fix_preview is None
    assert len(linted_only.fix_conflict_preview()) == 0

    # Fixed file with precomputed patches/preview removed builds from tree.
    linted_fixed = linter.lint_string(sql, fix=True)
    legacy = linted_fixed._replace(source_patches=None, fix_preview=None)
    fallback = legacy.fix_conflict_preview()
    assert len(fallback) == 1
    assert fallback.edits[0].origins == ["source patches"]


def test__linted_file__preview_across_jinja_variants_is_compatible():
    """Fixes found in separate render branches map to disjoint source ranges.

    A real Jinja ``{% if %}`` renders twice (root + uncovered branch); each
    branch's LT01 fix maps back to its own source range, so they are
    compatible and both origins are recorded. The edits apply together and
    match the merged result used by the normal fix path.
    """
    sql = "{% if true %}\nselect  1 from a\n{% else %}\nselect  2 from b\n{% endif %}\n"
    config = FluffConfig(
        overrides={"dialect": "ansi", "rules": "LT01", "templater": "jinja"}
    )
    linted = Linter(config=config).lint_string(sql, fix=True)

    preview = linted.fix_conflict_preview()
    assert len(preview) == 2
    assert [edit.origins[0] for edit in preview] == [
        "root_variant",
        "alternate_variant_1",
    ]
    assert all(edit.relation is EditRelation.COMPATIBLE for edit in preview)

    fixed = linted.apply_fix_selection([edit.patch for edit in preview])
    assert "select 1 from a" in fixed
    assert "select 2 from b" in fixed
    fix_string, changed = linted.fix_string()
    assert changed
    assert fixed == fix_string


def test__linted_file__preview_dedupes_shared_literal_across_variants():
    """The same edit found in two renders of one source range is IDENTICAL.

    The literal after ``{% endif %}`` is rendered in every variant. Both
    variants report the LT01 fix for it, so the preview shows a single edit
    suggested by both origins, applied only once.
    """
    sql = (
        "{% if true %}\n"
        "select 1 from a\n"
        "{% else %}\n"
        "select 2 from b\n"
        "{% endif %}\n"
        "from  x\n"
    )
    config = FluffConfig(
        overrides={"dialect": "ansi", "rules": "LT01", "templater": "jinja"}
    )
    linted = Linter(config=config).lint_string(sql, fix=True)

    preview = linted.fix_conflict_preview()
    assert len(preview) == 1
    edit = preview.edits[0]
    assert edit.source_str == "  "
    assert edit.fixed_raw == " "
    assert edit.relation is EditRelation.IDENTICAL
    assert edit.origins == ["root_variant", "alternate_variant_1"]

    assert linted.apply_fix_selection([edit.patch]) == sql.replace("from  x", "from x")

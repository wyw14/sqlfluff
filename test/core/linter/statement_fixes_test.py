"""Tests for per-statement fix preview and selective application."""

import pytest

from sqlfluff.core import Linter
from sqlfluff.core.config import FluffConfig
from sqlfluff.core.errors import SQLParseError
from sqlfluff.core.linter import (
    KIND_BOUNDARY,
    KIND_FILE,
    KIND_STATEMENT,
    KIND_TEMPLATE,
    StaleSourceError,
    StatementFixPreview,
    apply_statement_fixes,
    render_fixed_source,
)

TWO_STATEMENTS = "select  a from t ; select b  from t2 ;"
ADJACENT_STATEMENTS = "select a from t;select b from t2;"
JINJA_LOOP = (
    "{% for x in ['a','b'] %}select  {{ x }} from t"
    "{% if not loop.last %};{% endif %}\n{% endfor %}"
)


def _fixed_linted_file(sql, *, linter=None, fname="<string input>"):
    linter = linter or Linter(dialect="ansi")
    return linter, linter.lint_string(sql, fname=fname, fix=True)


def test_statement_preview_attributes_edits_per_statement():
    """Each edit is explicitly attributed to its owning statement."""
    linter, linted_file = _fixed_linted_file(TWO_STATEMENTS)
    preview = linted_file.statement_fix_preview()

    assert isinstance(preview, StatementFixPreview)
    assert len(preview.statements) == 2
    statement_zero, statement_one = preview.statements

    assert statement_zero.original_source == "select  a from t ;"
    assert statement_zero.fixed_source == "select a from t;"
    assert statement_one.original_source == "select b  from t2 ;"
    assert statement_one.fixed_source == "select b from t2;"

    statement_edits = [edit for edit in preview.edits if edit.kind == KIND_STATEMENT]
    assert {edit.statement_indices for edit in statement_edits} == {(0,), (1,)}
    # Every statement edit carries the rule code which produced it.
    assert all(edit.rule_codes for edit in statement_edits)
    # The trailing newline edit is outside both statements.
    file_edits = [edit for edit in preview.edits if edit.kind == KIND_FILE]
    assert file_edits and file_edits[0].fixed_source == "\n"
    assert file_edits[0].statement_indices == (0, 1)


def test_selecting_one_statement_leaves_other_untouched():
    """Unselected statements must remain byte-for-byte unchanged."""
    linter, linted_file = _fixed_linted_file(TWO_STATEMENTS)
    result = linter.apply_statement_fixes(linted_file, [0])

    assert result.success
    assert result.fixed_source == "select a from t; select b  from t2 ;"
    assert "select b  from t2 ;" in result.fixed_source
    # The file-level trailing newline edit is held back too.
    assert not result.fixed_source.endswith("\n")
    assert len(result.applied_edits) == 2
    assert len(result.held_edits) == 3
    # The result of a partial selection must still parse.
    assert result.violations == ()


def test_full_selection_matches_full_fix_string():
    """Selecting every statement reproduces the ordinary fix output."""
    linter, linted_file = _fixed_linted_file(TWO_STATEMENTS)
    expected, changed = linted_file.fix_string()
    assert changed

    result = linter.apply_statement_fixes(linted_file, [0, 1])
    assert result.success
    assert result.fixed_source == expected
    assert result.held_edits == ()

    # Selecting nothing is a no-op which returns the original source.
    noop = linter.apply_statement_fixes(linted_file, [])
    assert noop.success
    assert noop.fixed_source == TWO_STATEMENTS
    assert noop.applied_edits == ()


def test_boundary_edit_requires_both_statements():
    """An edit between two statements is not given to either one."""
    linter, linted_file = _fixed_linted_file(ADJACENT_STATEMENTS)
    preview = linted_file.statement_fix_preview()

    boundary_edits = [
        edit for edit in preview.shared_edits if edit.kind == KIND_BOUNDARY
    ]
    assert len(boundary_edits) == 1
    assert boundary_edits[0].statement_indices == (0, 1)
    assert boundary_edits[0].fixed_source == " "

    # Neither statement alone can claim the boundary whitespace.
    assert render_fixed_source(preview, [0]) == ADJACENT_STATEMENTS
    assert render_fixed_source(preview, [1]) == ADJACENT_STATEMENTS
    # Selecting both applies it.
    assert render_fixed_source(preview, [0, 1]) == linted_file.fix_string()[0]


def test_file_level_edit_requires_all_statements():
    """Leading/trailing padding edits are held until everything is selected."""
    sql = "\n\nselect a from t;\nselect b from t;\n\n"
    linter, linted_file = _fixed_linted_file(sql)
    preview = linted_file.statement_fix_preview()

    assert {edit.kind for edit in preview.shared_edits} == {KIND_FILE}
    assert render_fixed_source(preview, [0]) == sql
    assert render_fixed_source(preview, [1]) == sql
    fully_fixed = render_fixed_source(preview, [0, 1])
    assert fully_fixed == linted_file.fix_string()[0]
    assert fully_fixed == "select a from t;\nselect b from t;\n"


def test_shared_template_fragment_requires_every_rendered_statement():
    """Source in a Jinja loop renders into several statements: shared edit."""
    config = FluffConfig.from_root(overrides={"dialect": "ansi", "templater": "jinja"})
    linter = Linter(config=config)
    linted_file = linter.lint_string(JINJA_LOOP, fname="loop.sql", fix=True)
    preview = linted_file.statement_fix_preview()

    template_edits = [
        edit
        for edit in preview.shared_edits
        if edit.kind in (KIND_TEMPLATE, "boundary_and_template")
    ]
    assert len(template_edits) == 1
    assert template_edits[0].statement_indices == (0, 1)
    assert template_edits[0].original_source == "  "

    # The edit cannot be applied to just one rendered statement.
    partial = linter.apply_statement_fixes(linted_file, [0])
    assert partial.success
    assert "select  {{ x }}" in partial.fixed_source
    assert partial.applied_edits == ()

    full = linter.apply_statement_fixes(linted_file, [0, 1])
    assert full.success
    assert full.fixed_source == linted_file.fix_string()[0]
    assert "select {{ x }}" in full.fixed_source


def test_application_is_rejected_when_source_changed():
    """Stale offsets must never overwrite newer source content."""
    linter, linted_file = _fixed_linted_file(TWO_STATEMENTS)
    with pytest.raises(StaleSourceError) as exc_info:
        linter.apply_statement_fixes(
            linted_file,
            [0],
            current_source=TWO_STATEMENTS + "\n-- edited since preview\n",
        )
    assert exc_info.value.path == linted_file.path
    # The hashes of both the preview and the mismatched source are reported.
    assert exc_info.value.expected_hash != exc_info.value.actual_hash


def test_invalid_statement_indices_are_rejected():
    """Selecting an index which does not exist fails fast."""
    linter, linted_file = _fixed_linted_file(TWO_STATEMENTS)
    preview = linted_file.statement_fix_preview()
    with pytest.raises(ValueError, match="Unknown statement indices"):
        render_fixed_source(preview, [2])


def test_failed_reparse_blocks_application():
    """When the selection does not reparse, success is False and nothing held.

    The reparse checker is injected here to simulate a selection which
    produces unparsable SQL.
    """
    linter, linted_file = _fixed_linted_file(TWO_STATEMENTS)
    preview = linted_file.statement_fix_preview()

    def broken_parse_checker(_source):
        return [SQLParseError("Rendered selection does not parse")]

    result = apply_statement_fixes(
        preview,
        [0],
        current_source=TWO_STATEMENTS,
        parse_checker=broken_parse_checker,
    )
    assert not result.success
    assert len(result.violations) == 1
    assert result.path is None


def test_selective_persist_roundtrip_and_stale_file_guard(tmp_path):
    """End-to-end file workflow: selective persist plus disk-change guard."""
    path = tmp_path / "multi.sql"
    path.write_text(TWO_STATEMENTS, encoding="utf-8")
    linter = Linter(dialect="ansi")
    linted_dir = linter.lint_path(str(path), fix=True)
    linted_file = linted_dir.files[0]

    result = linter.apply_statement_fixes(linted_file, [0], persist=True)
    assert result.success
    assert result.path == str(path)

    on_disk = path.read_text(encoding="utf-8")
    assert on_disk == result.fixed_source
    assert on_disk.startswith("select a from t;")
    # The second statement was not touched.
    assert "select b  from t2 ;" in on_disk

    # Change the file out of band: the preview offsets are now stale and
    # applying them must be refused without touching the file.
    path.write_text("-- a brand new file\nSELECT 1;", encoding="utf-8")
    with pytest.raises(StaleSourceError):
        linter.apply_statement_fixes(linted_file, [0, 1], persist=True)
    assert path.read_text(encoding="utf-8") == "-- a brand new file\nSELECT 1;"


def test_clean_file_preview_has_no_edits():
    """A file which is already clean yields an empty, applicable preview."""
    linter = Linter(dialect="ansi")
    linted_file = linter.lint_string("select a from t;\n", fix=True)
    preview = linted_file.statement_fix_preview()
    assert not preview.has_changes
    assert preview.edits == []
    assert render_fixed_source(preview, [0]) == "select a from t;\n"


def test_preview_without_parsed_tree_fails():
    """A preview requires templating and parsing to have succeeded."""
    config = FluffConfig.from_root(overrides={"dialect": "ansi", "templater": "jinja"})
    linter = Linter(config=config)
    linted_file = linter.lint_string(
        "{% for %}broken template", fname="broken.sql", fix=True
    )
    assert linted_file.tree is None
    with pytest.raises(ValueError, match="templated and parsed"):
        linted_file.statement_fix_preview()

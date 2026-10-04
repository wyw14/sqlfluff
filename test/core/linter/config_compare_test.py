"""Tests for comparing lint findings across two rule configurations."""

import json

import pytest

from sqlfluff.core import FluffConfig, Linter
from sqlfluff.core.linter.config_compare import (
    DeltaReason,
    ViolationView,
    build_alias_map,
    compare_configs,
    format_comparison_human,
    match_violations,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _view(
    code: str = "LT01",
    line_no: int = 1,
    line_pos: int = 1,
    source_start: int | None = None,
    source_end: int | None = None,
    anchor_raw: str | None = "x",
    description: str = "some issue",
    warning: bool = False,
    fix_signature: tuple = (),
    failure_kind: str | None = None,
) -> ViolationView:
    return ViolationView(
        code=code,
        canonical_code=code,
        description=description,
        warning=warning,
        line_no=line_no,
        line_pos=line_pos,
        source_start=source_start,
        source_end=source_end,
        anchor_raw=anchor_raw,
        fix_signature=fix_signature,
        failure_kind=failure_kind,
        record={"code": code, "description": description},
    )


def _config(config_text: str) -> FluffConfig:
    return FluffConfig.from_string(config_text)


# ---------------------------------------------------------------------------
# Rule alias handling
# ---------------------------------------------------------------------------


def test_build_alias_map_contains_legacy_codes() -> None:
    """Legacy rule codes map onto their current canonical codes."""
    alias_map = build_alias_map()
    assert alias_map["L011"] == "AL01"
    assert alias_map["L014"] == "CP02"
    # Canonical codes are never keys of the alias map.
    assert "AL01" not in alias_map
    assert "CP02" not in alias_map


# ---------------------------------------------------------------------------
# Structural matching
# ---------------------------------------------------------------------------


def test_match_follows_issue_that_moved_position() -> None:
    """Same code/anchor/message still matches when the position moves."""
    baseline = [_view(line_no=3, line_pos=5, source_start=20, source_end=21)]
    candidate = [_view(line_no=4, line_pos=9, source_start=30, source_end=31)]
    pairs, disappeared, new = match_violations(baseline, candidate)
    assert len(pairs) == 1
    assert pairs[0].position_changed is True
    assert disappeared == [] and new == []


def test_match_does_not_pair_unrelated_findings() -> None:
    """Different anchor and message at a different position don't match."""
    baseline = [_view(anchor_raw="a", description="first", line_no=1, line_pos=1)]
    candidate = [_view(anchor_raw="b", description="second", line_no=2, line_pos=5)]
    pairs, disappeared, new = match_violations(baseline, candidate)
    assert pairs == []
    assert len(disappeared) == 1 and len(new) == 1


def test_match_same_position_matches_even_if_message_changes() -> None:
    """The same rule at the identical source spot matches despite text changes."""
    baseline = [_view(source_start=8, source_end=9, description="old wording")]
    candidate = [_view(source_start=8, source_end=9, description="new wording")]
    pairs, _, _ = match_violations(baseline, candidate)
    assert len(pairs) == 1
    assert pairs[0].description_changed is True


def test_match_is_one_to_one_with_repeated_anchors() -> None:
    """Repeated anchor text is assigned to the nearest partner."""
    baseline = [
        _view(line_no=1, line_pos=1, source_start=0, source_end=1),
        _view(line_no=1, line_pos=10, source_start=9, source_end=10),
    ]
    candidate = [
        _view(line_no=1, line_pos=2, source_start=1, source_end=2),
        _view(line_no=1, line_pos=11, source_start=10, source_end=11),
    ]
    pairs, _, _ = match_violations(baseline, candidate)
    assert len(pairs) == 2
    paired_starts = {
        (pair.baseline.source_start, pair.candidate.source_start) for pair in pairs
    }
    assert paired_starts == {(0, 1), (9, 10)}


def test_match_never_crosses_rule_codes() -> None:
    """Findings from different rules are never paired."""
    baseline = [
        _view(code="AL01", source_start=0, source_end=1, description="d"),
    ]
    candidate = [
        _view(code="CP02", source_start=0, source_end=1, description="d"),
    ]
    pairs, disappeared, new = match_violations(baseline, candidate)
    assert pairs == []
    assert {v.canonical_code for v in disappeared} == {"AL01"}
    assert {v.canonical_code for v in new} == {"CP02"}


# ---------------------------------------------------------------------------
# End to end comparisons (stdin keeps full string-built configs)
# ---------------------------------------------------------------------------


_BASE_EXCLUDES = "exclude_rules = LT09,LT12,AM04,AM06"


def _only(result, filepath: str = "stdin", side: str | None = None):
    """Convenience accessor for the single file record."""
    assert len(result.files) == 1
    comparison = result.files[0]
    assert comparison.filepath == filepath
    return comparison


def test_compare_rule_disabled_and_enabled() -> None:
    """Disabling a rule removes its findings; enabling adds them."""
    sql = "SELECT * FROM users u\n"
    config_a = _config(f"[sqlfluff]\ndialect = ansi\n{_BASE_EXCLUDES}\n")
    config_b = _config(f"[sqlfluff]\ndialect = ansi\n{_BASE_EXCLUDES},AL01\n")

    # Baseline -> candidate: AL01 switched off.
    result = compare_configs(config_a, config_b, stdin_sql=sql)
    comparison = _only(result)
    assert comparison.comparable is True
    assert {item.pair.baseline.canonical_code for item in comparison.persisted} == {
        "AL05"
    }
    assert comparison.new == []
    disappeared = comparison.disappeared
    assert [item.view.canonical_code for item in disappeared] == ["AL01"]
    assert disappeared[0].reason == DeltaReason.RULE_DISABLED
    assert "AL01" in result.rules_disabled

    # The reverse comparison reports a newly enabled rule.
    reverse = compare_configs(config_b, config_a, stdin_sql=sql)
    reverse_comparison = _only(reverse)
    assert reverse_comparison.disappeared == []
    assert [item.view.canonical_code for item in reverse_comparison.new] == ["AL01"]
    assert reverse_comparison.new[0].reason == DeltaReason.RULE_ENABLED
    assert "AL01" in reverse.rules_enabled


def test_compare_rule_parameter_change() -> None:
    """Changed rule parameters flip which findings appear."""
    sql = "SELECT A, b FROM foo\n"
    config_a = _config(
        "[sqlfluff]\ndialect = ansi\nrules = CP02\n"
        "[sqlfluff:rules:capitalisation.identifiers]\n"
        "extended_capitalisation_policy = upper\n"
    )
    config_b = _config(
        "[sqlfluff]\ndialect = ansi\nrules = CP02\n"
        "[sqlfluff:rules:capitalisation.identifiers]\n"
        "extended_capitalisation_policy = lower\n"
    )
    result = compare_configs(config_a, config_b, stdin_sql=sql)
    comparison = _only(result)
    assert comparison.persisted == []
    new = comparison.new
    disappeared = comparison.disappeared
    assert {item.view.line_pos for item in new} == {8}
    assert {item.view.line_pos for item in disappeared} == {11, 18}
    assert all(item.reason == DeltaReason.RULE_PARAMETER_CHANGED for item in new)
    assert all(
        item.reason == DeltaReason.RULE_PARAMETER_CHANGED for item in disappeared
    )
    assert all(
        "extended_capitalisation_policy" in item.parameter_changes
        for item in new + disappeared
    )
    assert result.rule_parameter_changes["CP02"] == {
        "extended_capitalisation_policy": {"baseline": "upper", "candidate": "lower"}
    }
    # Enablement is identical on both sides.
    assert result.rules_enabled == set()
    assert result.rules_disabled == set()


def test_compare_undetermined_reason_for_dialect_change() -> None:
    """A finding driven by a non-rule config change is 'undetermined'."""
    sql = "SELECT help3 = 'hello'\n"
    section = "[sqlfluff:rules:tsql.prefer_as_alias]\nforce_enable = True\n"
    config_a = _config(f"[sqlfluff]\ndialect = ansi\nrules = TQ04\n{section}")
    config_b = _config(f"[sqlfluff]\ndialect = tsql\nrules = TQ04\n{section}")
    result = compare_configs(config_a, config_b, stdin_sql=sql)
    comparison = _only(result)
    assert comparison.new and not comparison.disappeared
    entry = comparison.new[0]
    assert entry.view.canonical_code == "TQ04"
    assert entry.reason == DeltaReason.UNDETERMINED
    other_paths = {
        item["path"] for item in result.as_records()["summary"]["other_config_changes"]
    }
    assert "core.dialect" in other_paths


def test_compare_identical_configs_everything_persists() -> None:
    """Two identical configs produce only persisted findings."""
    sql = "SELECT * FROM users u\n"
    config = _config(f"[sqlfluff]\ndialect = ansi\n{_BASE_EXCLUDES}\n")
    result = compare_configs(config, config.copy(), stdin_sql=sql)
    comparison = _only(result)
    assert comparison.comparable is True
    assert comparison.new == [] and comparison.disappeared == []
    assert {item.pair.baseline.canonical_code for item in comparison.persisted} == {
        "AL01",
        "AL05",
    }
    assert result.as_records()["summary"]["counts"]["persisted"] == 2


def test_compare_templating_failure_blocks_comparison() -> None:
    """A one-sided templating failure is shown and never reads as 'fixed'."""
    sql = "SELECT {{ x }} AS c FROM t\n"
    config_a = _config(
        "[sqlfluff]\ndialect = ansi\n[sqlfluff:templater:jinja:context]\nx = 1\n"
    )
    config_b = _config("[sqlfluff]\ndialect = ansi\n")
    result = compare_configs(config_a, config_b, stdin_sql=sql)
    comparison = _only(result)
    assert comparison.comparable is False
    assert "candidate" in (comparison.not_comparable_reason or "")
    assert comparison.baseline_failures == []
    failure_kinds = [view.failure_kind for view in comparison.candidate_failures]
    assert "templating" in failure_kinds
    # Nothing is labelled new or fixed when a failure is involved.
    assert comparison.new == [] and comparison.disappeared == []
    assert comparison.persisted == []

    human = format_comparison_human(result, nocolor=True)
    assert "NOT COMPARED" in human
    assert "templating" in human


def test_compare_parse_failure_both_sides_is_reported_separately() -> None:
    """Parsing failures on both sides are listed per side, not paired."""
    sql = "SELECT a FROM\n"
    config = _config("[sqlfluff]\ndialect = ansi\n")
    result = compare_configs(config, config.copy(), stdin_sql=sql)
    comparison = _only(result)
    assert comparison.comparable is False
    assert [v.failure_kind for v in comparison.baseline_failures] == ["parsing"]
    assert [v.failure_kind for v in comparison.candidate_failures] == ["parsing"]
    assert comparison.new == [] and comparison.disappeared == []


def test_compare_does_not_mutate_or_leak_configs() -> None:
    """Each side keeps an independent config and nothing is mutated."""
    sql = "SELECT * FROM users u\n"
    config_a = _config(f"[sqlfluff]\ndialect = ansi\n{_BASE_EXCLUDES}\n")
    config_b = _config(f"[sqlfluff]\ndialect = ansi\n{_BASE_EXCLUDES},AL01\n")
    a_denylist_before = list(config_a.get_section("core")["rule_denylist"])

    compare_configs(config_a, config_b, stdin_sql=sql)

    # The baseline config is untouched by the candidate's exclusion.
    assert list(config_a.get_section("core")["rule_denylist"]) == a_denylist_before
    # A fresh lint with the baseline config still finds AL01.
    fresh = Linter(config=config_a).lint_string(sql, fname="stdin")
    assert "AL01" in [v.rule_code() for v in fresh.get_violations(filter_warning=False)]
    # A fresh lint with the candidate config does not.
    fresh_b = Linter(config=config_b).lint_string(sql, fname="stdin")
    assert "AL01" not in [
        v.rule_code() for v in fresh_b.get_violations(filter_warning=False)
    ]


def test_compare_file_mode_is_readonly(tmp_path) -> None:
    """Path-based comparison lints identical content and never writes."""
    config_path_a = tmp_path / "old.sqlfluff"
    config_path_b = tmp_path / "new.sqlfluff"
    sql_path = tmp_path / "query.sql"
    config_path_a.write_text(
        f"[sqlfluff]\ndialect = ansi\n{_BASE_EXCLUDES}\n", encoding="utf-8"
    )
    config_path_b.write_text(
        f"[sqlfluff]\ndialect = ansi\n{_BASE_EXCLUDES},AL01\n",
        encoding="utf-8",
    )
    sql_path.write_text("SELECT * FROM users u\n", encoding="utf-8")

    config_a = FluffConfig.from_root(
        extra_config_path=str(config_path_a), ignore_local_config=True
    )
    config_b = FluffConfig.from_root(
        extra_config_path=str(config_path_b), ignore_local_config=True
    )
    before = sql_path.read_bytes()
    stat_before = sql_path.stat().st_mtime_ns

    result = compare_configs(config_a, config_b, (str(sql_path),))

    assert sql_path.read_bytes() == before
    assert sql_path.stat().st_mtime_ns == stat_before
    comparison = _only(result, filepath=str(sql_path))
    assert comparison.comparable is True
    assert [item.view.canonical_code for item in comparison.disappeared] == ["AL01"]
    assert comparison.disappeared[0].reason == DeltaReason.RULE_DISABLED


def test_compare_records_are_json_serialisable() -> None:
    """The structured output must round-trip through JSON."""
    sql = "SELECT * FROM users u\n"
    config_a = _config(f"[sqlfluff]\ndialect = ansi\n{_BASE_EXCLUDES}\n")
    config_b = _config(f"[sqlfluff]\ndialect = ansi\n{_BASE_EXCLUDES},AL01\n")
    result = compare_configs(config_a, config_b, stdin_sql=sql)
    records = result.as_records()
    json.loads(json.dumps(records))
    assert records["baseline"] == {"label": "baseline"}
    assert records["files"][0]["filepath"] == "stdin"


def test_compare_stdin_uses_filename_label(tmp_path) -> None:
    """A stdin filename is reflected in the file record."""
    target = tmp_path / "virtual.sql"
    target.write_text("SELECT a\n")
    # Overrides are inherited by the child config built for the stdin
    # path, mirroring how the CLI resolves --stdin-filename.
    config = FluffConfig.from_string("", overrides={"dialect": "ansi"})
    result = compare_configs(
        config,
        config.copy(),
        stdin_sql="SELECT a FROM b\n",
        stdin_filename=str(target),
    )
    assert result.files[0].filepath == str(target)


@pytest.mark.parametrize(
    "sql,expected_codes",
    [
        ("SELECT a FROM\n", {"PRS"}),
    ],
)
def test_human_output_contains_all_sections(sql, expected_codes) -> None:
    """The human rendering surfaces failures and the comparison buckets."""
    config = _config("[sqlfluff]\ndialect = ansi\n")
    result = compare_configs(config, config.copy(), stdin_sql=sql)
    text = format_comparison_human(result, nocolor=True)
    assert "Baseline failures" in text
    assert "Candidate failures" in text
    assert "NOT COMPARED" in text
    for code in expected_codes:
        assert code in text

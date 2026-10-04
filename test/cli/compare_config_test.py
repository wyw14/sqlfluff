"""Tests for the `sqlfluff compare-config` CLI command."""

import json

from click.testing import CliRunner

from sqlfluff.cli.commands import compare_config

_EXCLUDES = "exclude_rules = LT09,LT12,AM04,AM06"


def _write_configs(tmp_path, baseline_excludes=_EXCLUDES, candidate_excludes=None):
    candidate_excludes = candidate_excludes or f"{_EXCLUDES},AL01"
    baseline = tmp_path / "old.sqlfluff"
    candidate = tmp_path / "new.sqlfluff"
    baseline.write_text(
        f"[sqlfluff]\ndialect = ansi\n{baseline_excludes}\n"
        "[sqlfluff:rules:capitalisation.identifiers]\n"
        "extended_capitalisation_policy = upper\n",
        encoding="utf-8",
    )
    candidate.write_text(
        f"[sqlfluff]\ndialect = ansi\n{candidate_excludes}\n"
        "[sqlfluff:rules:capitalisation.identifiers]\n"
        "extended_capitalisation_policy = lower\n",
        encoding="utf-8",
    )
    return baseline, candidate


def test_compare_config_human_file_mode(tmp_path) -> None:
    """Human output reports the buckets and never touches the source file."""
    baseline, candidate = _write_configs(tmp_path)
    sql_path = tmp_path / "query.sql"
    sql_path.write_text("SELECT A, b FROM users u\n", encoding="utf-8")
    before = sql_path.read_bytes()

    runner = CliRunner()
    result = runner.invoke(
        compare_config,
        [
            "--baseline-config",
            str(baseline),
            "--candidate-config",
            str(candidate),
            "--ignore-local-config",
            str(sql_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Disabled by candidate: AL01" in result.stdout
    assert "rule_disabled" in result.stdout
    assert "rule_parameter_changed" in result.stdout
    assert "Persisted findings" in result.stdout
    assert "New in candidate" in result.stdout
    assert "Disappeared from baseline" in result.stdout
    # The comparison is read-only.
    assert sql_path.read_bytes() == before


def test_compare_config_json_stdin_real_files(tmp_path) -> None:
    """JSON output with real config files classifies every delta."""
    baseline, candidate = _write_configs(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        compare_config,
        [
            "--baseline-config",
            str(baseline),
            "--candidate-config",
            str(candidate),
            "--ignore-local-config",
            "--format",
            "json",
            "-",
        ],
        input="SELECT A, b FROM users u\n",
    )
    assert result.exit_code == 0, result.stdout
    records = json.loads(result.stdout)
    file_record = records["files"][0]
    assert file_record["comparable"] is True
    reasons_new = [item["reason"] for item in file_record["new"]]
    reasons_gone = [item["reason"] for item in file_record["disappeared"]]
    assert "rule_parameter_changed" in reasons_new
    assert "rule_disabled" in reasons_gone
    assert records["summary"]["rules_disabled"] == ["AL01"]
    assert "CP02" in records["summary"]["rule_parameter_changes"]


def test_compare_config_reports_templating_failure(tmp_path) -> None:
    """A one-sided templating failure is flagged and never counted fixed."""
    baseline = tmp_path / "ctx.sqlfluff"
    candidate = tmp_path / "noctx.sqlfluff"
    baseline.write_text(
        "[sqlfluff]\ndialect = ansi\n[sqlfluff:templater:jinja:context]\nx = 1\n",
        encoding="utf-8",
    )
    candidate.write_text("[sqlfluff]\ndialect = ansi\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        compare_config,
        [
            "--baseline-config",
            str(baseline),
            "--candidate-config",
            str(candidate),
            "--ignore-local-config",
            "-",
        ],
        input="SELECT {{ x }} AS c FROM t\n",
    )
    assert result.exit_code == 0, result.stdout
    assert "NOT COMPARED" in result.stdout
    assert "templating TMP" in result.stdout
    # Failure-driven absences are never presented as new/fixed buckets.
    assert "New in candidate" not in result.stdout
    assert "Disappeared from baseline" not in result.stdout


def test_compare_config_without_paths_errors(tmp_path) -> None:
    """At least one path (or stdin '-') is required."""
    baseline, candidate = _write_configs(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        compare_config,
        [
            "--baseline-config",
            str(baseline),
            "--candidate-config",
            str(candidate),
            "--ignore-local-config",
        ],
    )
    assert result.exit_code != 0
    assert "At least one path" in result.output


def test_compare_config_requires_both_configs(tmp_path) -> None:
    """Click rejects invocations missing a config side."""
    _, candidate = _write_configs(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        compare_config,
        ["--candidate-config", str(candidate), "-"],
        input="SELECT 1\n",
    )
    assert result.exit_code == 2
    assert "--baseline-config" in result.output


def test_compare_config_yaml_output(tmp_path) -> None:
    """YAML output is supported for machine consumption."""
    import yaml

    baseline, candidate = _write_configs(tmp_path)
    sql_path = tmp_path / "query.sql"
    sql_path.write_text("SELECT A, b FROM users u\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        compare_config,
        [
            "--baseline-config",
            str(baseline),
            "--candidate-config",
            str(candidate),
            "--ignore-local-config",
            "--format",
            "yaml",
            str(sql_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    records = yaml.safe_load(result.stdout)
    assert records["summary"]["files"] == 1


def test_compare_config_write_output(tmp_path) -> None:
    """The --write-output option redirects the payload to a file."""
    baseline, candidate = _write_configs(tmp_path)
    sql_path = tmp_path / "query.sql"
    sql_path.write_text("SELECT A, b FROM users u\n", encoding="utf-8")
    out_path = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(
        compare_config,
        [
            "--baseline-config",
            str(baseline),
            "--candidate-config",
            str(candidate),
            "--ignore-local-config",
            "--format",
            "json",
            "--write-output",
            str(out_path),
            str(sql_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    records = json.loads(out_path.read_text(encoding="utf-8"))
    assert records["files"][0]["filepath"] == str(sql_path)

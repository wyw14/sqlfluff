"""Compare lint results produced under two rule configurations.

The purpose of this module is to support upgrading (or otherwise changing)
the project rule configuration: the *same* source file content and the
*same* templater input is linted twice, once with a baseline config and
once with a candidate config, and the resulting findings are then matched
against each other.

Design guarantees:

* The two sides are fully independent: separate :class:`FluffConfig` and
  :class:`Linter` instances are constructed and neither config object is
  ever shared or mutated by the other side.
* Files are read (at most once per side) but never written or fixed, so
  comparison mode does not modify anything on disk.
* Templating, lexing and parsing failures are reported separately for
  each side. A file that fails on either side is marked as not
  comparable, so a missing finding caused by such a failure is never
  presented as a fixed or newly introduced issue.
* Findings are matched structurally (canonical rule code, source slice,
  anchor text, description and proposed fixes), not by error text, so
  small position moves and rule aliases are handled sensibly.
* New or disappeared findings are attributed to a rule being
  enabled/disabled, a rule parameter change, or an undetermined cause.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from colorama import Fore, Style

from sqlfluff.core.config.fluffconfig import FluffConfig
from sqlfluff.core.errors import (
    SQLFluffSkipFile,
    SQLLexError,
    SQLParseError,
    SQLTemplaterError,
)
from sqlfluff.core.helpers.dict import NestedStringDict
from sqlfluff.core.linter.discovery import paths_from_path
from sqlfluff.core.linter.linted_file import LintedFile
from sqlfluff.core.linter.linter import Linter
from sqlfluff.core.rules import get_ruleset

# Failure kinds keyed by the error type which produced them.
_FAILURE_KINDS = (
    (SQLTemplaterError, "templating"),
    (SQLParseError, "parsing"),
    (SQLLexError, "lexing"),
)

# Config paths which are represented directly by rule enabled/disabled
# status, and so should not also be listed as "other" config changes.
_ALLOWLIST_PATHS = {
    ("core", "rules"),
    ("core", "exclude_rules"),
    ("core", "rule_allowlist"),
    ("core", "rule_denylist"),
}

# Pair score at or above which two findings are considered the same issue.
# See `_pair_score` for how the score is constructed.
_MATCH_THRESHOLD = 45

# How much anchor text to retain for display / matching.
_ANCHOR_PREVIEW = 60


class DeltaReason:
    """Reasons why a finding only exists on one side of a comparison."""

    RULE_ENABLED = "rule_enabled"
    RULE_DISABLED = "rule_disabled"
    RULE_PARAMETER_CHANGED = "rule_parameter_changed"
    UNDETERMINED = "undetermined"


# ---------------------------------------------------------------------------
# Normalised view of a violation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ViolationView:
    """Structural view of a single finding, independent of error text."""

    code: str
    canonical_code: str
    description: str
    warning: bool
    line_no: int
    line_pos: int
    source_start: Optional[int]
    source_end: Optional[int]
    anchor_raw: Optional[str]
    fix_signature: tuple[Any, ...]
    failure_kind: Optional[str]
    record: dict[str, Any]

    @property
    def is_failure(self) -> bool:
        """Whether this view represents a templating/lexing/parsing failure."""
        return self.failure_kind is not None


def _failure_kind(error: Any) -> Optional[str]:
    for error_type, kind in _FAILURE_KINDS:
        if isinstance(error, error_type):
            return kind
    return None


def _fix_signature(error: Any) -> tuple[Any, ...]:
    fixes = getattr(error, "fixes", None) or []
    return tuple(
        tuple(edit.raw for edit in lint_fix.edit) if lint_fix.edit else None
        for lint_fix in fixes
    )


def build_violation_view(error: Any, alias_map: dict[str, str]) -> ViolationView:
    """Construct a normalised :class:`ViolationView` from an error object."""
    segment = getattr(error, "segment", None)
    pos_marker = getattr(segment, "pos_marker", None)
    source_start: Optional[int] = None
    source_end: Optional[int] = None
    anchor_raw: Optional[str] = None
    if pos_marker is not None:
        source_slice = getattr(pos_marker, "source_slice", None)
        if source_slice is not None:
            source_start = source_slice.start
            source_end = source_slice.stop
        source_str = getattr(pos_marker, "source_str", None)
        if source_str is not None:
            anchor_raw = source_str()[:_ANCHOR_PREVIEW]
    if anchor_raw is None and segment is not None:
        raw = getattr(segment, "raw", None)
        if raw is not None:
            anchor_raw = raw[:_ANCHOR_PREVIEW]
    code = error.rule_code()
    record = dict(error.to_dict())
    return ViolationView(
        code=code,
        canonical_code=alias_map.get(code, code),
        description=error.desc(),
        warning=bool(getattr(error, "warning", False)),
        line_no=error.line_no,
        line_pos=error.line_pos,
        source_start=source_start,
        source_end=source_end,
        anchor_raw=anchor_raw,
        fix_signature=_fix_signature(error),
        failure_kind=_failure_kind(error),
        record=record,
    )


# ---------------------------------------------------------------------------
# Rule reference (alias) handling
# ---------------------------------------------------------------------------


def build_alias_map() -> dict[str, str]:
    """Map any rule alias/name to its single canonical rule code.

    References which resolve to multiple codes (groups) are excluded.
    Reported findings always carry canonical codes, but this defends the
    comparison against legacy/alias codes and documents the intent.
    """
    reference_map = get_ruleset().rule_reference_map()
    alias_map: dict[str, str] = {}
    for ref, codes in reference_map.items():
        if len(codes) == 1:
            canonical = next(iter(codes))
            if ref != canonical:
                alias_map[ref] = canonical
    return alias_map


# ---------------------------------------------------------------------------
# Finding matching
# ---------------------------------------------------------------------------


def _pair_score(left: ViolationView, right: ViolationView) -> Optional[int]:
    """Score how likely two findings are to be the same underlying issue."""
    if left.canonical_code != right.canonical_code:
        return None
    score = 0
    # Source slice is the strongest signal: the source file content is
    # identical on both sides, so the same logical issue usually anchors
    # at the same source offset.
    if (
        left.source_start is not None
        and right.source_start is not None
        and left.source_start == right.source_start
        and left.source_end == right.source_end
    ):
        score += 50
    elif (
        left.source_start is not None
        and right.source_start is not None
        and left.source_end is not None
        and right.source_end is not None
        and left.source_start < right.source_end
        and right.source_start < left.source_end
    ):
        # Overlapping (but not identical) source ranges.
        score += 30
    elif left.line_no == right.line_no and left.line_pos == right.line_pos:
        score += 20
    else:
        # Fallback proximity, used to order otherwise-tied candidates.
        if left.line_no == right.line_no:
            score += 12
        elif abs(left.line_no - right.line_no) <= 3:
            score += 8
        elif abs(left.line_no - right.line_no) <= 10:
            score += 4
    # The anchor token text identifies the flagged source element, which
    # lets us follow issues that have moved to a different position.
    if left.anchor_raw is not None and left.anchor_raw == right.anchor_raw:
        score += 25
    if left.description == right.description:
        score += 20
    if left.fix_signature and left.fix_signature == right.fix_signature:
        score += 15
    return score


@dataclass(frozen=True)
class MatchedPair:
    """A finding which exists on both sides of the comparison."""

    baseline: ViolationView
    candidate: ViolationView
    score: int

    @property
    def position_changed(self) -> bool:
        """Whether the reported location differs."""
        return (
            self.baseline.line_no != self.candidate.line_no
            or self.baseline.line_pos != self.candidate.line_pos
            or self.baseline.source_start != self.candidate.source_start
            or self.baseline.source_end != self.candidate.source_end
        )

    @property
    def description_changed(self) -> bool:
        """Whether the issue message differs."""
        return self.baseline.description != self.candidate.description

    @property
    def anchor_changed(self) -> bool:
        """Whether the flagged anchor text differs."""
        return self.baseline.anchor_raw != self.candidate.anchor_raw

    @property
    def warning_changed(self) -> bool:
        """Whether the warning/failure status of the finding differs."""
        return self.baseline.warning != self.candidate.warning

    @property
    def fixes_changed(self) -> bool:
        """Whether the proposed fixes differ."""
        return self.baseline.fix_signature != self.candidate.fix_signature

    def change_notes(self) -> list[str]:
        """Human readable notes about what changed for a persisted finding."""
        notes: list[str] = []
        if self.position_changed:
            notes.append(
                "position moved "
                f"{self.baseline.line_no}:{self.baseline.line_pos} -> "
                f"{self.candidate.line_no}:{self.candidate.line_pos}"
            )
        if self.anchor_changed:
            notes.append("anchor changed")
        if self.description_changed:
            notes.append("message changed")
        if self.fixes_changed:
            notes.append("fix changed")
        if self.warning_changed:
            notes.append(
                "now a warning" if self.candidate.warning else "no longer a warning"
            )
        return notes


def match_violations(
    baseline: list[ViolationView], candidate: list[ViolationView]
) -> tuple[list[MatchedPair], list[ViolationView], list[ViolationView]]:
    """Match findings across the two sides.

    Returns the matched pairs together with the unmatched baseline
    (disappeared) and unmatched candidate (new) findings.

    Matching is a deterministic greedy maximum-weight assignment: all
    candidate pairs above the match threshold are considered, the
    strongest pairs are assigned first and each finding is used at most
    once.
    """
    candidates: list[tuple[int, int, int, int, int]] = []
    for i, left in enumerate(baseline):
        for j, right in enumerate(candidate):
            score = _pair_score(left, right)
            if score is None or score < _MATCH_THRESHOLD:
                continue
            candidates.append(
                (
                    score,
                    abs(left.line_no - right.line_no),
                    abs(left.line_pos - right.line_pos),
                    i,
                    j,
                )
            )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))

    pairs: list[MatchedPair] = []
    used_baseline: set[int] = set()
    used_candidate: set[int] = set()
    for score, _, _, i, j in candidates:
        if i in used_baseline or j in used_candidate:
            continue
        used_baseline.add(i)
        used_candidate.add(j)
        pairs.append(MatchedPair(baseline[i], candidate[j], score))

    unmatched_baseline = [
        view for i, view in enumerate(baseline) if i not in used_baseline
    ]
    unmatched_candidate = [
        view for j, view in enumerate(candidate) if j not in used_candidate
    ]
    pairs.sort(
        key=lambda pair: (
            pair.baseline.line_no,
            pair.baseline.line_pos,
            pair.baseline.canonical_code,
        )
    )
    return pairs, unmatched_baseline, unmatched_candidate


# ---------------------------------------------------------------------------
# Config comparison (rule enablement and rule parameters)
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Coerce config values into JSON-serialisable builtin types."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return repr(value)


def _flatten_leaves(
    configs: NestedStringDict[Any], prefix: tuple[str, ...] = ()
) -> dict[tuple[str, ...], Any]:
    """Flatten a config mapping to leaf paths."""
    leaves: dict[tuple[str, ...], Any] = {}
    for key, value in configs.items():
        path = prefix + (str(key),)
        if str(key).endswith("_obj"):
            # Generated objects (dialect_obj, templater_obj) are derived
            # from other settings and are not meaningful to compare.
            continue
        if isinstance(value, dict):
            leaves.update(_flatten_leaves(value, path))
        else:
            leaves[path] = value
    return leaves


@dataclass
class RuleConfigDelta:
    """Effective rule-configuration difference between the two sides."""

    enabled: set[str] = field(default_factory=set)
    disabled: set[str] = field(default_factory=set)
    # code -> parameter path -> {"baseline": v, "candidate": v}
    parameter_changes: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=dict
    )
    # {"path": "section.key", "baseline": v, "candidate": v}
    other_changes: list[dict[str, Any]] = field(default_factory=list)


def compare_rule_config(
    config_a: FluffConfig,
    config_b: FluffConfig,
    pack_a: Any,
    pack_b: Any,
) -> RuleConfigDelta:
    """Work out which rule switches and parameter changes divide two packs.

    ``pack_a`` and ``pack_b`` are the effective :class:`RulePack` objects
    for the file being compared (i.e. they already reflect any nested or
    inline configuration).
    """
    rules_a = {rule.code: rule for rule in pack_a.rules}
    rules_b = {rule.code: rule for rule in pack_b.rules}
    codes_a = set(rules_a)
    codes_b = set(rules_b)

    delta = RuleConfigDelta(
        enabled=codes_b - codes_a,
        disabled=codes_a - codes_b,
    )

    # Map config refs (rule names) and generic rule config keywords back
    # to canonical rule codes.
    ref_to_codes: dict[str, set[str]] = {}
    generic_keyword_codes: dict[str, set[str]] = {}
    for code, rule in {**rules_a, **rules_b}.items():
        ref_to_codes.setdefault(rule.get_config_ref(), set()).add(code)
        for keyword in rule.config_keywords:
            generic_keyword_codes.setdefault(keyword, set()).add(code)

    leaves_a = _flatten_leaves(config_a._configs)
    leaves_b = _flatten_leaves(config_b._configs)
    all_paths = set(leaves_a) | set(leaves_b)
    for path in sorted(all_paths):
        value_a = leaves_a.get(path)
        value_b = leaves_b.get(path)
        if value_a == value_b:
            continue
        if path in _ALLOWLIST_PATHS:
            # Represented directly by the enabled/disabled sets.
            continue
        target_codes: set[str]
        parameter_path: Optional[str]
        if len(path) >= 3 and path[0] == "rules" and path[1] in ref_to_codes:
            # A setting inside a rule's own section, e.g.
            # ("rules", "capitalisation.identifiers", "capitalisation_policy").
            target_codes = ref_to_codes[path[1]]
            parameter_path = ".".join(path[2:])
        elif len(path) == 2 and path[0] == "rules" and path[1] in generic_keyword_codes:
            # A generic rule keyword which is only meaningful to rules
            # that declare it, e.g. ("rules", "capitalisation_policy").
            target_codes = generic_keyword_codes[path[1]]
            parameter_path = path[1]
        else:
            delta.other_changes.append(
                {
                    "path": ".".join(path),
                    "baseline": _json_safe(value_a),
                    "candidate": _json_safe(value_b),
                }
            )
            continue
        for code in target_codes:
            # Only attribute a parameter change to rules which exist on
            # both sides; rules appearing/disappearing are covered by
            # the enabled/disabled sets.
            if code not in codes_a or code not in codes_b:
                continue
            delta.parameter_changes.setdefault(code, {})[parameter_path] = {
                "baseline": _json_safe(value_a),
                "candidate": _json_safe(value_b),
            }
    return delta


def _classify_reason(
    code: str, codes_a: set[str], codes_b: set[str], delta: RuleConfigDelta
) -> str:
    if code in codes_b and code not in codes_a:
        return DeltaReason.RULE_ENABLED
    if code in codes_a and code not in codes_b:
        return DeltaReason.RULE_DISABLED
    if delta.parameter_changes.get(code):
        return DeltaReason.RULE_PARAMETER_CHANGED
    return DeltaReason.UNDETERMINED


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class DeltaViolation:
    """A finding which only exists on one side, with an attributed reason."""

    view: ViolationView
    reason: str
    parameter_changes: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the one-sided finding."""
        record = dict(self.view.record)
        record["canonical_code"] = self.view.canonical_code
        if self.view.anchor_raw is not None:
            record["anchor_raw"] = self.view.anchor_raw
        return {
            "reason": self.reason,
            "parameter_changes": _json_safe(self.parameter_changes),
            "violation": record,
        }


@dataclass
class PersistedViolation:
    """A matched finding, plus details of how it changed."""

    pair: MatchedPair
    parameter_changes: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the matched finding pair."""
        baseline = dict(self.pair.baseline.record)
        candidate = dict(self.pair.candidate.record)
        baseline["canonical_code"] = self.pair.baseline.canonical_code
        candidate["canonical_code"] = self.pair.candidate.canonical_code
        if self.pair.baseline.anchor_raw is not None:
            baseline["anchor_raw"] = self.pair.baseline.anchor_raw
        if self.pair.candidate.anchor_raw is not None:
            candidate["anchor_raw"] = self.pair.candidate.anchor_raw
        return {
            "baseline": baseline,
            "candidate": candidate,
            "changes": self.pair.change_notes(),
            "parameter_changes": _json_safe(self.parameter_changes),
        }


def _failure_record(view: ViolationView) -> dict[str, Any]:
    return {
        "kind": view.failure_kind,
        "code": view.code,
        "description": view.description,
        "start_line_no": view.line_no,
        "start_line_pos": view.line_pos,
    }


@dataclass
class FileComparison:
    """The comparison outcome for a single file."""

    filepath: str
    comparable: bool
    not_comparable_reason: Optional[str]
    baseline_failures: list[ViolationView]
    candidate_failures: list[ViolationView]
    baseline_skipped: Optional[str]
    candidate_skipped: Optional[str]
    persisted: list[PersistedViolation] = field(default_factory=list)
    new: list[DeltaViolation] = field(default_factory=list)
    disappeared: list[DeltaViolation] = field(default_factory=list)
    # Findings shown but not matched when a side failed/skipped. These are
    # deliberately never labelled as fixed or new.
    baseline_unmatched_blocked: list[ViolationView] = field(default_factory=list)
    candidate_unmatched_blocked: list[ViolationView] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable record for the file comparison."""
        record: dict[str, Any] = {
            "filepath": self.filepath,
            "comparable": self.comparable,
            "not_comparable_reason": self.not_comparable_reason,
            "baseline": {
                "skipped": self.baseline_skipped,
                "failures": [_failure_record(v) for v in self.baseline_failures],
            },
            "candidate": {
                "skipped": self.candidate_skipped,
                "failures": [_failure_record(v) for v in self.candidate_failures],
            },
        }
        if self.comparable:
            record["persisted"] = [item.as_record() for item in self.persisted]
            record["new"] = [item.as_record() for item in self.new]
            record["disappeared"] = [item.as_record() for item in self.disappeared]
        else:
            record["baseline"]["violations_not_compared"] = [
                _failure_like_record(v) for v in self.baseline_unmatched_blocked
            ]
            record["candidate"]["violations_not_compared"] = [
                _failure_like_record(v) for v in self.candidate_unmatched_blocked
            ]
        return record


def _failure_like_record(view: ViolationView) -> dict[str, Any]:
    record = dict(view.record)
    record["canonical_code"] = view.canonical_code
    if view.anchor_raw is not None:
        record["anchor_raw"] = view.anchor_raw
    return record


@dataclass
class ComparisonResult:
    """The full result of comparing two rule configurations."""

    baseline_label: str
    candidate_label: str
    files: list[FileComparison]
    rules_enabled: set[str] = field(default_factory=set)
    rules_disabled: set[str] = field(default_factory=set)
    rule_parameter_changes: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=dict
    )
    other_config_changes: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_records(self) -> dict[str, Any]:
        """Return a JSON/YAML-serialisable representation of the result."""
        persisted_count = sum(len(item.persisted) for item in self.files)
        new_count = sum(len(item.new) for item in self.files)
        disappeared_count = sum(len(item.disappeared) for item in self.files)
        baseline_failures = sum(len(item.baseline_failures) for item in self.files)
        candidate_failures = sum(len(item.candidate_failures) for item in self.files)
        return {
            "baseline": {"label": self.baseline_label},
            "candidate": {"label": self.candidate_label},
            "summary": {
                "files": len(self.files),
                "files_comparable": sum(1 for item in self.files if item.comparable),
                "rules_enabled": sorted(self.rules_enabled),
                "rules_disabled": sorted(self.rules_disabled),
                "rule_parameter_changes": _json_safe(self.rule_parameter_changes),
                "other_config_changes": [
                    {"path": path, **values}
                    for path, values in sorted(self.other_config_changes.items())
                ],
                "counts": {
                    "persisted": persisted_count,
                    "new": new_count,
                    "disappeared": disappeared_count,
                    "baseline_failures": baseline_failures,
                    "candidate_failures": candidate_failures,
                },
            },
            "files": [item.as_record() for item in self.files],
        }


# ---------------------------------------------------------------------------
# Per-file comparison
# ---------------------------------------------------------------------------


def _sorted_views(views: list[ViolationView]) -> list[ViolationView]:
    return sorted(
        views,
        key=lambda view: (
            view.line_no,
            view.line_pos,
            view.canonical_code,
            view.description,
        ),
    )


def _compare_linted_files(
    filepath: str,
    linted_a: Optional[LintedFile],
    linted_b: Optional[LintedFile],
    file_config_a: Optional[FluffConfig],
    file_config_b: Optional[FluffConfig],
    linter_a: Linter,
    linter_b: Linter,
    alias_map: dict[str, str],
    skipped_a: Optional[str],
    skipped_b: Optional[str],
) -> tuple[FileComparison, RuleConfigDelta]:
    """Compare two linted versions of the same file content."""
    failures_a: list[ViolationView] = []
    failures_b: list[ViolationView] = []
    findings_a: list[ViolationView] = []
    findings_b: list[ViolationView] = []

    def partition(
        linted: Optional[LintedFile],
        failures: list[ViolationView],
        findings: list[ViolationView],
    ) -> None:
        if linted is None:
            return
        for error in linted.get_violations(filter_warning=False):
            view = build_violation_view(error, alias_map)
            if view.is_failure:
                failures.append(view)
            else:
                findings.append(view)

    partition(linted_a, failures_a, findings_a)
    partition(linted_b, failures_b, findings_b)
    failures_a = _sorted_views(failures_a)
    failures_b = _sorted_views(failures_b)

    blocked_reasons: list[str] = []
    if skipped_a:
        blocked_reasons.append(f"baseline skipped the file: {skipped_a}")
    if skipped_b:
        blocked_reasons.append(f"candidate skipped the file: {skipped_b}")
    if failures_a:
        blocked_reasons.append("baseline produced templating/lexing/parsing failures")
    if failures_b:
        blocked_reasons.append("candidate produced templating/lexing/parsing failures")

    comparison = FileComparison(
        filepath=filepath,
        comparable=not blocked_reasons,
        not_comparable_reason=("; ".join(blocked_reasons) if blocked_reasons else None),
        baseline_failures=failures_a,
        candidate_failures=failures_b,
        baseline_skipped=skipped_a,
        candidate_skipped=skipped_b,
    )

    # We still inspect the effective rule packs/configuration whenever
    # possible so that the run-level summary remains informative even for
    # files which failed to render on one side.
    pack_a = linter_a.get_rulepack(config=file_config_a) if file_config_a else None
    pack_b = linter_b.get_rulepack(config=file_config_b) if file_config_b else None
    if pack_a is not None and pack_b is not None and file_config_a and file_config_b:
        delta = compare_rule_config(file_config_a, file_config_b, pack_a, pack_b)
    else:
        delta = RuleConfigDelta()

    if blocked_reasons:
        # Do not conclude "new" or "fixed": surface findings separately so
        # a failure can never masquerade as a resolved issue.
        comparison.baseline_unmatched_blocked = _sorted_views(findings_a)
        comparison.candidate_unmatched_blocked = _sorted_views(findings_b)
        return comparison, delta

    # Both sides rendered successfully, so both packs must be available.
    assert pack_a is not None and pack_b is not None
    codes_a = {rule.code for rule in pack_a.rules}
    codes_b = {rule.code for rule in pack_b.rules}

    pairs, unmatched_a, unmatched_b = match_violations(
        _sorted_views(findings_a), _sorted_views(findings_b)
    )
    for pair in pairs:
        comparison.persisted.append(
            PersistedViolation(
                pair=pair,
                parameter_changes=delta.parameter_changes.get(
                    pair.baseline.canonical_code, {}
                ),
            )
        )
    for view in unmatched_b:
        reason = _classify_reason(view.canonical_code, codes_a, codes_b, delta)
        comparison.new.append(
            DeltaViolation(
                view=view,
                reason=reason,
                parameter_changes=delta.parameter_changes.get(view.canonical_code, {})
                if reason == DeltaReason.RULE_PARAMETER_CHANGED
                else {},
            )
        )
    for view in unmatched_a:
        reason = _classify_reason(view.canonical_code, codes_a, codes_b, delta)
        comparison.disappeared.append(
            DeltaViolation(
                view=view,
                reason=reason,
                parameter_changes=delta.parameter_changes.get(view.canonical_code, {})
                if reason == DeltaReason.RULE_PARAMETER_CHANGED
                else {},
            )
        )
    return comparison, delta


# ---------------------------------------------------------------------------
# Top level entry point
# ---------------------------------------------------------------------------


def _load_side(
    linter: Linter, config: FluffConfig, fname: str
) -> tuple[Optional[str], Optional[FluffConfig], Optional[str], Optional[str]]:
    """Load raw content + effective config for one side.

    Returns ``(raw, file_config, encoding, skip_reason)``.
    """
    try:
        raw, file_config, encoding = linter.load_raw_file_and_config(fname, config)
    except SQLFluffSkipFile as exc:
        return None, None, None, str(exc)
    return raw, file_config, encoding, None


def _lint_side(
    linter: Linter,
    raw: str,
    fname: str,
    file_config: FluffConfig,
    encoding: str,
) -> tuple[Optional[LintedFile], Optional[str]]:
    try:
        return (
            linter.lint_string(
                raw,
                fname=fname,
                config=file_config,
                encoding=encoding,
                fix=False,
            ),
            None,
        )
    except SQLFluffSkipFile as exc:
        return None, str(exc)


def compare_configs(
    config_a: FluffConfig,
    config_b: FluffConfig,
    paths: tuple[str, ...] = (),
    *,
    stdin_sql: Optional[str] = None,
    stdin_filename: Optional[str] = None,
    ignore_files: bool = True,
    baseline_label: Optional[str] = None,
    candidate_label: Optional[str] = None,
) -> ComparisonResult:
    """Lint the same inputs under two configs and compare the findings.

    Args:
        config_a: The baseline (e.g. current) configuration.
        config_b: The candidate (e.g. upgraded) configuration.
        paths: SQL file or directory paths to lint on both sides.
        stdin_sql: When comparing content passed on standard input, the
            exact content to compare. Mutually exclusive with ``paths``
            discovery; both sides receive this identical string.
        stdin_filename: Optional filename used to resolve configuration
            for the stdin input.
        ignore_files: Whether to honour ``.sqlfluffignore`` files when
            discovering paths.
        baseline_label: Human readable label for the baseline side.
        candidate_label: Human readable label for the candidate side.

    Returns:
        A :class:`ComparisonResult` containing the structured comparison.

    The two configurations are used independently (each side gets its own
    :class:`Linter`) and no file is ever written to.
    """
    linter_a = Linter(config=config_a)
    linter_b = Linter(config=config_b)
    alias_map = build_alias_map()
    result = ComparisonResult(
        baseline_label=baseline_label or "baseline",
        candidate_label=candidate_label or "candidate",
        files=[],
    )

    def _record(
        filepath: str,
        raw_a: Optional[str],
        raw_b: Optional[str],
        fca: Optional[FluffConfig],
        fcb: Optional[FluffConfig],
        encoding_a: Optional[str],
        encoding_b: Optional[str],
        skip_a: Optional[str],
        skip_b: Optional[str],
    ) -> None:
        content_mismatch = raw_a is not None and raw_b is not None and raw_a != raw_b
        linted_a: Optional[LintedFile] = None
        linted_b: Optional[LintedFile] = None
        if raw_a is not None and fca is not None and encoding_a is not None:
            linted_a, lint_skip_a = _lint_side(
                linter_a, raw_a, filepath, fca, encoding_a
            )
            skip_a = skip_a or lint_skip_a
        if raw_b is not None and fcb is not None and encoding_b is not None:
            linted_b, lint_skip_b = _lint_side(
                linter_b, raw_b, filepath, fcb, encoding_b
            )
            skip_b = skip_b or lint_skip_b
        comparison, delta = _compare_linted_files(
            filepath,
            linted_a,
            linted_b,
            fca,
            fcb,
            linter_a,
            linter_b,
            alias_map,
            skip_a,
            skip_b,
        )
        if content_mismatch and comparison.comparable:
            # The two configs interpreted the on-disk bytes differently
            # (e.g. different encodings). The inputs were not identical,
            # so refuse to draw new/fixed conclusions. The findings were
            # already partitioned inside the comparison; move them into
            # the separate "not compared" buckets and discard any pairs.
            comparison.comparable = False
            comparison.not_comparable_reason = (
                "file content decoded differently under the two configs "
                "(e.g. different encoding settings)"
            )
            blocked_a = [item.pair.baseline for item in comparison.persisted]
            blocked_a.extend(item.view for item in comparison.disappeared)
            blocked_b = [item.pair.candidate for item in comparison.persisted]
            blocked_b.extend(item.view for item in comparison.new)
            comparison.baseline_unmatched_blocked = _sorted_views(blocked_a)
            comparison.candidate_unmatched_blocked = _sorted_views(blocked_b)
            comparison.persisted = []
            comparison.new = []
            comparison.disappeared = []
        result.files.append(comparison)
        _merge_summary(result, delta)

    if stdin_sql is not None:
        fca = (
            config_a.make_child_from_path(stdin_filename, require_dialect=False)
            if stdin_filename
            else config_a.copy()
        )
        fcb = (
            config_b.make_child_from_path(stdin_filename, require_dialect=False)
            if stdin_filename
            else config_b.copy()
        )
        fca.process_raw_file_for_config(stdin_sql, "stdin")
        fcb.process_raw_file_for_config(stdin_sql, "stdin")
        _record(
            stdin_filename or "stdin",
            stdin_sql,
            stdin_sql,
            fca,
            fcb,
            "utf-8",
            "utf-8",
            None,
            None,
        )
        result.files.sort(key=lambda comparison: comparison.filepath)
        return result

    sql_exts = config_a.get("sql_file_exts", default=".sql").lower().split(",")
    fnames: list[str] = []
    for path in paths:
        fnames.extend(
            paths_from_path(
                path,
                ignore_non_existent_files=False,
                ignore_files=ignore_files,
                target_file_exts=sql_exts,
            )
        )

    for fname in fnames:
        raw_a, file_config_a, encoding_a, skip_a = _load_side(linter_a, config_a, fname)
        raw_b, file_config_b, encoding_b, skip_b = _load_side(linter_b, config_b, fname)
        # Lint the identical source string on both sides. If only one
        # side failed to load, its findings are reported via the skip
        # reason; the other side is still rendered but can't be compared.
        raw_for_a = raw_a if raw_a is not None else raw_b
        raw_for_b = raw_b if raw_b is not None else raw_a
        _record(
            fname,
            raw_for_a,
            raw_for_b,
            file_config_a,
            file_config_b,
            encoding_a or encoding_b,
            encoding_b or encoding_a,
            skip_a,
            skip_b,
        )
    result.files.sort(key=lambda comparison: comparison.filepath)
    return result


def _merge_summary(result: ComparisonResult, delta: RuleConfigDelta) -> None:
    """Merge a per-file config delta into the run-level summary."""
    result.rules_enabled.update(delta.enabled)
    result.rules_disabled.update(delta.disabled)
    for code, parameters in delta.parameter_changes.items():
        result.rule_parameter_changes.setdefault(code, {})
        for parameter, values in parameters.items():
            result.rule_parameter_changes[code].setdefault(parameter, values)
    for change in delta.other_changes:
        result.other_config_changes.setdefault(
            change["path"],
            {"baseline": change["baseline"], "candidate": change["candidate"]},
        )


# ---------------------------------------------------------------------------
# Human readable rendering
# ---------------------------------------------------------------------------


def _location(view: ViolationView) -> str:
    return f"L{view.line_no}:{view.line_pos}"


def _format_violation_line(view: ViolationView, prefix: str = "") -> str:
    warning = " (warning)" if view.warning else ""
    return (
        f"    {prefix}{view.canonical_code}  {_location(view)}  "
        f"{view.description}{warning}"
    )


def format_comparison_human(result: ComparisonResult, nocolor: bool = False) -> str:
    """Render a :class:`ComparisonResult` as human readable text."""

    def color(text: str, colour: str) -> str:
        if nocolor:
            return text
        return f"{colour}{text}{Style.RESET_ALL}"

    lines: list[str] = []
    lines.append(
        color(
            f"=== Rule configuration comparison: {result.baseline_label} "
            f"vs {result.candidate_label} ===",
            Fore.CYAN,
        )
    )

    # Configuration summary.
    lines.append("")
    lines.append("Configuration changes:")
    if result.rules_enabled:
        lines.append(
            "  Enabled by candidate : " + ", ".join(sorted(result.rules_enabled))
        )
    if result.rules_disabled:
        lines.append(
            "  Disabled by candidate: " + ", ".join(sorted(result.rules_disabled))
        )
    if result.rule_parameter_changes:
        lines.append("  Rule parameters changed:")
        for code in sorted(result.rule_parameter_changes):
            for parameter, values in sorted(
                result.rule_parameter_changes[code].items()
            ):
                lines.append(
                    f"    {code}.{parameter}: "
                    f"{values['baseline']!r} -> {values['candidate']!r}"
                )
    if result.other_config_changes:
        lines.append("  Other config changes (not attributable to a rule):")
        for path, values in sorted(result.other_config_changes.items()):
            lines.append(
                f"    {path}: {values['baseline']!r} -> {values['candidate']!r}"
            )
    if not (
        result.rules_enabled
        or result.rules_disabled
        or result.rule_parameter_changes
        or result.other_config_changes
    ):
        lines.append("  (none)")

    for comparison in result.files:
        lines.append("")
        lines.append(color(f"=== File: {comparison.filepath} ===", Fore.CYAN))

        lines.append(f"  Baseline failures ({result.baseline_label}):")
        if comparison.baseline_skipped:
            lines.append(f"    skipped: {comparison.baseline_skipped}")
        if comparison.baseline_failures:
            for view in comparison.baseline_failures:
                lines.append(
                    f"    [{view.failure_kind} {view.code}] {_location(view)} "
                    f"{view.description}"
                )
        elif not comparison.baseline_skipped:
            lines.append("    (none)")
        lines.append(f"  Candidate failures ({result.candidate_label}):")
        if comparison.candidate_skipped:
            lines.append(f"    skipped: {comparison.candidate_skipped}")
        if comparison.candidate_failures:
            for view in comparison.candidate_failures:
                lines.append(
                    f"    [{view.failure_kind} {view.code}] {_location(view)} "
                    f"{view.description}"
                )
        elif not comparison.candidate_skipped:
            lines.append("    (none)")

        if not comparison.comparable:
            lines.append("")
            lines.append(
                color(
                    "  NOT COMPARED: "
                    f"{comparison.not_comparable_reason}. "
                    "Findings are shown separately below; they are NOT counted "
                    "as new or fixed.",
                    Fore.RED,
                )
            )
            if comparison.baseline_unmatched_blocked:
                lines.append(
                    f"  Baseline findings not compared "
                    f"({len(comparison.baseline_unmatched_blocked)}):"
                )
                for view in comparison.baseline_unmatched_blocked:
                    lines.append(_format_violation_line(view))
            if comparison.candidate_unmatched_blocked:
                lines.append(
                    f"  Candidate findings not compared "
                    f"({len(comparison.candidate_unmatched_blocked)}):"
                )
                for view in comparison.candidate_unmatched_blocked:
                    lines.append(_format_violation_line(view))
            continue

        lines.append(f"  Persisted findings ({len(comparison.persisted)}):")
        if comparison.persisted:
            for item in comparison.persisted:
                line = _format_violation_line(item.pair.candidate)
                notes = item.pair.change_notes()
                if item.parameter_changes:
                    notes.append(
                        "rule parameter changed: "
                        + ", ".join(sorted(item.parameter_changes))
                    )
                if notes:
                    line += "  [" + "; ".join(notes) + "]"
                lines.append(line)
        else:
            lines.append("    (none)")

        lines.append(
            color(
                f"  New in candidate ({len(comparison.new)}):",
                Fore.LIGHTYELLOW_EX,
            )
        )
        if comparison.new:
            for delta in comparison.new:
                detail = ""
                if delta.reason == DeltaReason.RULE_PARAMETER_CHANGED:
                    detail = " (" + ", ".join(sorted(delta.parameter_changes)) + ")"
                lines.append(
                    _format_violation_line(
                        delta.view, prefix=f"[{delta.reason}{detail}] "
                    )
                )
        else:
            lines.append("    (none)")

        lines.append(
            color(
                f"  Disappeared from baseline ({len(comparison.disappeared)}):",
                Fore.GREEN,
            )
        )
        if comparison.disappeared:
            for delta in comparison.disappeared:
                detail = ""
                if delta.reason == DeltaReason.RULE_PARAMETER_CHANGED:
                    detail = " (" + ", ".join(sorted(delta.parameter_changes)) + ")"
                lines.append(
                    _format_violation_line(
                        delta.view, prefix=f"[{delta.reason}{detail}] "
                    )
                )
        else:
            lines.append("    (none)")

    return "\n".join(lines)

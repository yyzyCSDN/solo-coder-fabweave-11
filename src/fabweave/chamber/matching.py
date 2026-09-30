from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from fabweave.core.units import finite_non_negative
from fabweave.core.versioning import RevisionHistory

MATCHED = "matched"
MISMATCHED = "mismatched"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

_VERDICTS = (MATCHED, MISMATCHED, INSUFFICIENT_EVIDENCE)


def matched(reference: Mapping[str, float], candidate: Mapping[str, float], tolerances: Mapping[str, float]) -> bool:
    return not mismatch_reasons(reference, candidate, {key: Tolerance(absolute=value) for key, value in tolerances.items()})


@dataclass(frozen=True, slots=True)
class Tolerance:
    absolute: float = 0.0
    relative: float = 0.0

    def __post_init__(self) -> None:
        finite_non_negative(self.absolute, "absolute tolerance")
        finite_non_negative(self.relative, "relative tolerance")


def mismatch_reasons(
    reference: Mapping[str, float],
    candidate: Mapping[str, float],
    tolerances: Mapping[str, Tolerance],
) -> tuple[str, ...]:
    reasons: list[str] = []
    for key, tolerance in sorted(tolerances.items()):
        if key not in reference or key not in candidate:
            reasons.append(f"{key}:missing")
            continue
        expected, actual = reference[key], candidate[key]
        try:
            finite_non_negative(expected, f"{key} reference")
            finite_non_negative(actual, f"{key} candidate")
        except ValueError:
            reasons.append(f"{key}:non_finite_or_negative")
            continue
        limit = max(tolerance.absolute, abs(expected) * tolerance.relative)
        if abs(expected - actual) > limit:
            reasons.append(f"{key}:outside_tolerance")
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class ChamberSample:
    """One multi-parameter metrology reading from a single chamber run."""

    sample_id: str
    measured_at: int
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id is required")
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True, slots=True)
class SamplePair:
    golden: ChamberSample
    candidate: ChamberSample

    @property
    def compared_at(self) -> int:
        """A pair can only be judged once both readings exist."""
        return max(self.golden.measured_at, self.candidate.measured_at)

    @property
    def label(self) -> str:
        return f"{self.golden.sample_id}~{self.candidate.sample_id}"


@dataclass(frozen=True, slots=True)
class PairingResult:
    pairs: tuple[SamplePair, ...]
    unpaired_golden: tuple[str, ...]
    unpaired_candidate: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PairOutcome:
    pair: SamplePair
    rule_version: int
    deltas: Mapping[str, float]
    violations: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "deltas", MappingProxyType(dict(self.deltas)))


@dataclass(frozen=True, slots=True)
class UnevaluablePair:
    pair: SamplePair
    reason: str


@dataclass(frozen=True, slots=True)
class MatchReport:
    verdict: str
    pairs: tuple[PairOutcome, ...]
    unevaluable: tuple[UnevaluablePair, ...]
    unpaired_golden: tuple[str, ...]
    unpaired_candidate: tuple[str, ...]
    rule_versions: tuple[int, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.verdict not in _VERDICTS:
            raise ValueError("unknown verdict")
        if self.verdict == MATCHED and self.reasons:
            raise ValueError("a matched report cannot carry reasons")
        if self.verdict != MATCHED and not self.reasons:
            raise ValueError("a non-matched report requires reasons")


def _canonical_order(samples: list[ChamberSample], chamber: str) -> list[ChamberSample]:
    ordered = sorted(samples, key=lambda sample: (sample.measured_at, sample.sample_id))
    counts: dict[str, int] = {}
    for sample in ordered:
        counts[sample.sample_id] = counts.get(sample.sample_id, 0) + 1
    duplicates = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {chamber} sample ids: {', '.join(duplicates)}")
    return ordered


def pair_samples(golden: list[ChamberSample], candidate: list[ChamberSample], max_gap: int) -> PairingResult:
    """Pair each golden reading with the nearest still-unpaired candidate within ``max_gap``.

    Both chambers are processed in canonical (measured_at, sample_id) order and ties
    break on (gap, measured_at, sample_id), so the pairing depends only on the input
    set, never on the order samples were supplied in.
    """

    finite_non_negative(max_gap, "max_gap")
    goldens = _canonical_order(golden, "golden")
    remaining = _canonical_order(candidate, "candidate")
    pairs: list[SamplePair] = []
    for reference in goldens:
        best: ChamberSample | None = None
        best_key: tuple[int, int, str] | None = None
        for reading in remaining:
            gap = abs(reading.measured_at - reference.measured_at)
            if gap > max_gap:
                continue
            key = (gap, reading.measured_at, reading.sample_id)
            if best_key is None or key < best_key:
                best, best_key = reading, key
        if best is not None:
            pairs.append(SamplePair(reference, best))
            remaining.remove(best)
    paired_golden = {pair.golden.sample_id for pair in pairs}
    return PairingResult(
        tuple(pairs),
        tuple(sample.sample_id for sample in goldens if sample.sample_id not in paired_golden),
        tuple(sample.sample_id for sample in remaining),
    )


def paired_match_report(
    golden: list[ChamberSample],
    candidate: list[ChamberSample],
    rules: RevisionHistory[dict[str, Tolerance]],
    max_gap: int,
    min_pairs: int,
) -> MatchReport:
    """Compare chambers on paired multi-parameter samples instead of pooled means.

    Each pair is judged per parameter by the rule revision effective when both of its
    readings existed.  The verdict stays INSUFFICIENT_EVIDENCE unless at least
    ``min_pairs`` pairs were formed and every formed pair could be judged, so missing
    data or missing rule coverage never turns into a pass or a fail.
    """

    if min_pairs <= 0:
        raise ValueError("min_pairs must be positive")
    pairing = pair_samples(golden, candidate, max_gap)
    outcomes: list[PairOutcome] = []
    unevaluable: list[UnevaluablePair] = []
    for pair in pairing.pairs:
        try:
            revision = rules.at(pair.compared_at)
        except LookupError:
            unevaluable.append(UnevaluablePair(pair, "no_effective_rule"))
            continue
        violations = mismatch_reasons(pair.golden.values, pair.candidate.values, revision.value)
        deltas = {
            key: pair.candidate.values[key] - pair.golden.values[key]
            for key in sorted(revision.value)
            if key in pair.golden.values
            and key in pair.candidate.values
            and math.isfinite(pair.golden.values[key])
            and math.isfinite(pair.candidate.values[key])
        }
        outcomes.append(PairOutcome(pair, revision.version, deltas, violations))
    rule_versions = tuple(sorted({outcome.rule_version for outcome in outcomes}))
    reasons: list[str] = []
    if len(outcomes) < min_pairs:
        reasons.append(f"insufficient_pairs:{len(outcomes)}<{min_pairs}")
        if unevaluable:
            reasons.append(f"unevaluable_pairs:{len(unevaluable)}")
        verdict = INSUFFICIENT_EVIDENCE
    elif unevaluable:
        reasons.append(f"unevaluable_pairs:{len(unevaluable)}")
        verdict = INSUFFICIENT_EVIDENCE
    else:
        violations = [f"{outcome.pair.label}:{violation}" for outcome in outcomes for violation in outcome.violations]
        if violations:
            reasons.extend(violations)
            verdict = MISMATCHED
        else:
            verdict = MATCHED
    return MatchReport(
        verdict,
        tuple(outcomes),
        tuple(unevaluable),
        pairing.unpaired_golden,
        pairing.unpaired_candidate,
        rule_versions,
        tuple(reasons),
    )

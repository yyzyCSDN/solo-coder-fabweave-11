from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from fabweave.core.decision import Decision, allow, deny
from fabweave.core.timebox import TimeBox
from fabweave.core.units import finite_non_negative
from fabweave.core.versioning import RevisionHistory


def matched(reference: dict[str, float], candidate: dict[str, float], tolerances: dict[str, float]) -> bool:
    return not mismatch_reasons(reference, candidate, {key: Tolerance(absolute=value) for key, value in tolerances.items()})


@dataclass(frozen=True, slots=True)
class Tolerance:
    absolute: float = 0.0
    relative: float = 0.0

    def __post_init__(self) -> None:
        finite_non_negative(self.absolute, "absolute tolerance")
        finite_non_negative(self.relative, "relative tolerance")


def mismatch_reasons(
    reference: dict[str, float],
    candidate: dict[str, float],
    tolerances: dict[str, Tolerance],
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
    """A multi-parameter measurement taken on a chamber at a point in time."""

    sample_id: str
    at: int
    measurements: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        if self.at < 0:
            raise ValueError("sample time must be non-negative")
        object.__setattr__(self, "measurements", MappingProxyType(dict(self.measurements)))


def _pair_within_window(
    golden: Iterable[ChamberSample],
    candidate: Iterable[ChamberSample],
    window: TimeBox,
) -> tuple[tuple[ChamberSample, ChamberSample], ...]:
    """Pair golden and candidate samples that fall inside ``window``.

    Samples are sorted deterministically before pairing, so reordering the
    input lists never changes the result.  Pairing is one-to-one: each sample
    is used at most once, greedily matching the smallest time gap first.
    """

    golden_in = sorted(
        (sample for sample in golden if window.contains(sample.at)),
        key=lambda sample: (sample.at, sample.sample_id),
    )
    candidate_in = sorted(
        (sample for sample in candidate if window.contains(sample.at)),
        key=lambda sample: (sample.at, sample.sample_id),
    )
    edges = [
        (abs(golden.at - candidate.at), golden_index, candidate_index)
        for golden_index, golden in enumerate(golden_in)
        for candidate_index, candidate in enumerate(candidate_in)
    ]
    paired: list[tuple[ChamberSample, ChamberSample]] = []
    used_golden: set[int] = set()
    used_candidate: set[int] = set()
    for _, golden_index, candidate_index in sorted(edges):
        if golden_index in used_golden or candidate_index in used_candidate:
            continue
        used_golden.add(golden_index)
        used_candidate.add(candidate_index)
        paired.append((golden_in[golden_index], candidate_in[candidate_index]))
    paired.sort(key=lambda pair: (pair[0].at, pair[1].at, pair[0].sample_id, pair[1].sample_id))
    return tuple(paired)


def _judge_pair(
    golden: ChamberSample,
    candidate: ChamberSample,
    tolerances: Mapping[str, Tolerance],
) -> tuple[tuple[str, ...], dict[str, dict[str, object]]]:
    reasons: list[str] = []
    parameters: dict[str, dict[str, object]] = {}
    for key, tolerance in sorted(tolerances.items()):
        if key not in golden.measurements or key not in candidate.measurements:
            reasons.append(f"{key}:missing")
            parameters[key] = {"status": "missing"}
            continue
        expected = golden.measurements[key]
        actual = candidate.measurements[key]
        try:
            finite_non_negative(expected, f"{key} reference")
            finite_non_negative(actual, f"{key} candidate")
        except ValueError:
            reasons.append(f"{key}:non_finite_or_negative")
            parameters[key] = {"status": "non_finite_or_negative"}
            continue
        limit = max(tolerance.absolute, abs(expected) * tolerance.relative)
        deviation = abs(expected - actual)
        within = deviation <= limit
        parameters[key] = {
            "golden": expected,
            "candidate": actual,
            "limit": limit,
            "deviation": deviation,
            "within_tolerance": within,
        }
        if not within:
            reasons.append(f"{key}:outside_tolerance")
    return tuple(reasons), parameters


def match_chambers(
    golden: Iterable[ChamberSample],
    candidate: Iterable[ChamberSample],
    window: TimeBox,
    tolerances: RevisionHistory[Mapping[str, Tolerance]],
    min_pairs: int,
) -> Decision:
    """Match a golden chamber against a candidate chamber over a time window.

    Multi-parameter samples inside ``window`` are paired by nearest time and
    each pair is judged against the tolerance revision effective at the
    pair's time.  With fewer than ``min_pairs`` pairs the service fails
    closed: it reports ``insufficient_evidence`` and refuses to pass the
    candidate.  The report carries the member samples and the rule versions
    used; reordering the input samples does not change the result.
    """

    if min_pairs <= 0:
        raise ValueError("min_pairs must be positive")
    pairs = _pair_within_window(golden, candidate, window)
    governing = tolerances.latest().version

    member_samples: list[dict[str, object]] = []
    versions: set[int] = set()
    reasons: list[str] = []
    for golden_sample, candidate_sample in pairs:
        pair_time = max(golden_sample.at, candidate_sample.at)
        revision = tolerances.at(pair_time)
        versions.add(revision.version)
        pair_reasons, parameters = _judge_pair(golden_sample, candidate_sample, revision.value)
        member_samples.append(
            {
                "golden": golden_sample.sample_id,
                "candidate": candidate_sample.sample_id,
                "paired_at": pair_time,
                "rule_version": revision.version,
                "parameters": parameters,
            }
        )
        for reason in pair_reasons:
            reasons.append(f"pair:{golden_sample.sample_id}:{candidate_sample.sample_id}:{reason}")

    evidence: dict[str, object] = {
        "window": (window.start, window.end),
        "pair_count": len(pairs),
        "min_pairs": min_pairs,
        "member_samples": tuple(member_samples),
        "rule_versions": tuple(sorted(versions)) if versions else (governing,),
    }

    if len(pairs) < min_pairs:
        return deny("insufficient_evidence", **evidence)
    return deny(*reasons, **evidence) if reasons else allow(**evidence)

from __future__ import annotations

import pytest

from fabweave.chamber.matching import (
    INSUFFICIENT_EVIDENCE,
    MATCHED,
    MISMATCHED,
    ChamberSample,
    Tolerance,
    matched,
    pair_samples,
    paired_match_report,
)
from fabweave.core.versioning import RevisionHistory
from fabweave.workflows.control import chamber_pair_decision


def sample(sample_id: str, measured_at: int, **values: float) -> ChamberSample:
    return ChamberSample(sample_id, measured_at, values)


def rules(*revisions: tuple[int, int, dict[str, Tolerance]]) -> RevisionHistory[dict[str, Tolerance]]:
    history: RevisionHistory[dict[str, Tolerance]] = RevisionHistory()
    for version, effective_at, tolerances in revisions:
        history.publish(version, effective_at, tolerances)
    return history


def test_pairing_matches_within_window_and_reports_unpaired_members():
    golden = [sample("G1", 10, pressure=1.0), sample("G2", 20, pressure=1.0), sample("G3", 95, pressure=1.0)]
    candidate = [sample("C1", 12, pressure=1.0), sample("C2", 25, pressure=1.0), sample("C3", 97, pressure=1.0)]
    pairing = pair_samples(golden, candidate, 4)
    assert [(pair.golden.sample_id, pair.candidate.sample_id) for pair in pairing.pairs] == [("G1", "C1"), ("G3", "C3")]
    assert pairing.unpaired_golden == ("G2",)
    assert pairing.unpaired_candidate == ("C2",)


def test_pairing_prefers_nearest_and_breaks_ties_deterministically():
    golden = [sample("G", 10, pressure=1.0)]
    candidate = [sample("C_FAR", 14, pressure=1.0), sample("C_NEAR", 11, pressure=1.0)]
    (pair,) = pair_samples(golden, candidate, 5).pairs
    assert pair.candidate.sample_id == "C_NEAR"
    tied = [sample("C_B", 9, pressure=1.0), sample("C_A", 11, pressure=1.0)]
    (pair,) = pair_samples(golden, tied, 5).pairs
    assert pair.candidate.sample_id == "C_B"


def test_report_is_invariant_to_input_order():
    golden = [
        sample("G1", 0, pressure=10.0, flow=100.0),
        sample("G2", 10, pressure=11.0, flow=101.0),
        sample("G3", 20, pressure=12.0, flow=102.0),
    ]
    candidate = [
        sample("C1", 1, pressure=10.5, flow=105.0),
        sample("C2", 11, pressure=11.5, flow=95.0),
        sample("C3", 21, pressure=12.5, flow=100.0),
    ]
    history = rules((1, 0, {"pressure": Tolerance(absolute=1.0), "flow": Tolerance(relative=0.1)}))
    baseline = paired_match_report(golden, candidate, history, 4, 2)
    reordered = paired_match_report(list(reversed(golden)), [candidate[1], candidate[2], candidate[0]], history, 4, 2)
    assert baseline.verdict == MATCHED
    assert baseline == reordered


def test_paired_comparison_catches_drift_that_pooled_means_hide():
    golden = [sample("G1", 0, pressure=10.0), sample("G2", 10, pressure=20.0)]
    candidate = [sample("C1", 1, pressure=20.0), sample("C2", 11, pressure=10.0)]
    history = rules((1, 0, {"pressure": Tolerance(absolute=1.0)}))
    assert matched({"pressure": 15.0}, {"pressure": 15.0}, {"pressure": 1.0})
    report = paired_match_report(golden, candidate, history, 5, 2)
    assert report.verdict == MISMATCHED
    assert report.reasons == ("G1~C1:pressure:outside_tolerance", "G2~C2:pressure:outside_tolerance")
    assert [(outcome.pair.golden.sample_id, outcome.pair.candidate.sample_id) for outcome in report.pairs] == [("G1", "C1"), ("G2", "C2")]


def test_each_pair_is_judged_by_the_rule_effective_at_its_comparison_time():
    history = rules(
        (1, 0, {"pressure": Tolerance(absolute=2.0)}),
        (2, 100, {"pressure": Tolerance(absolute=0.5)}),
    )
    golden = [sample("G_OLD", 40, pressure=10.0), sample("G_NEW", 140, pressure=10.0)]
    candidate = [sample("C_OLD", 50, pressure=11.0), sample("C_NEW", 150, pressure=11.0)]
    report = paired_match_report(golden, candidate, history, 20, 2)
    assert report.verdict == MISMATCHED
    assert [outcome.rule_version for outcome in report.pairs] == [1, 2]
    assert report.rule_versions == (1, 2)
    assert report.reasons == ("G_NEW~C_NEW:pressure:outside_tolerance",)


def test_insufficient_pairs_refuses_to_conclude():
    history = rules((1, 0, {"pressure": Tolerance(absolute=2.0)}))
    report = paired_match_report([sample("G1", 0, pressure=10.0)], [sample("C1", 1, pressure=10.5)], history, 5, 2)
    assert report.verdict == INSUFFICIENT_EVIDENCE
    assert report.reasons == ("insufficient_pairs:1<2",)
    empty = paired_match_report([], [], history, 5, 1)
    assert empty.verdict == INSUFFICIENT_EVIDENCE
    assert empty.reasons == ("insufficient_pairs:0<1",)


def test_pairs_without_an_effective_rule_cannot_support_a_conclusion():
    history = rules((1, 100, {"pressure": Tolerance(absolute=2.0)}))
    golden = [sample("G_EARLY", 10, pressure=10.0), sample("G_LATE", 110, pressure=10.0)]
    candidate = [sample("C_EARLY", 12, pressure=10.5), sample("C_LATE", 112, pressure=10.5)]
    report = paired_match_report(golden, candidate, history, 5, 1)
    assert report.verdict == INSUFFICIENT_EVIDENCE
    assert report.reasons == ("unevaluable_pairs:1",)
    assert report.unevaluable[0].reason == "no_effective_rule"
    assert report.unevaluable[0].pair.golden.sample_id == "G_EARLY"
    starved = paired_match_report(golden[:1], candidate[:1], history, 5, 1)
    assert starved.reasons == ("insufficient_pairs:0<1", "unevaluable_pairs:1")


def test_missing_parameter_is_reported_not_silently_passed():
    history = rules((1, 0, {"pressure": Tolerance(absolute=1.0), "flow": Tolerance(absolute=5.0)}))
    golden = [sample("G1", 0, pressure=10.0, flow=100.0), sample("G2", 10, pressure=10.0, flow=100.0)]
    candidate = [sample("C1", 1, pressure=10.5), sample("C2", 11, pressure=10.25, flow=99.0)]
    report = paired_match_report(golden, candidate, history, 5, 2)
    assert report.verdict == MISMATCHED
    assert report.reasons == ("G1~C1:flow:missing",)
    assert report.pairs[0].violations == ("flow:missing",)
    assert report.pairs[0].deltas == {"pressure": 0.5}
    assert report.pairs[1].deltas == {"flow": -1.0, "pressure": 0.25}


def test_duplicate_sample_ids_and_bad_thresholds_are_rejected():
    history = rules((1, 0, {"pressure": Tolerance(absolute=1.0)}))
    with pytest.raises(ValueError):
        pair_samples([sample("G", 0, pressure=1.0), sample("G", 1, pressure=2.0)], [], 5)
    with pytest.raises(ValueError):
        pair_samples([], [], -1)
    with pytest.raises(ValueError):
        paired_match_report([], [], history, 5, 0)


def test_chamber_pair_decision_carries_member_samples_and_rule_versions():
    history = rules((1, 0, {"pressure": Tolerance(absolute=1.0)}))
    golden = [sample("G1", 0, pressure=10.0), sample("G2", 10, pressure=11.0)]
    candidate = [sample("C1", 1, pressure=10.5), sample("C2", 11, pressure=11.4)]
    decision = chamber_pair_decision(golden, candidate, history, 5, 2)
    assert decision.allowed
    assert decision.evidence["verdict"] == MATCHED
    assert decision.evidence["member_pairs"] == (("G1", "C1"), ("G2", "C2"))
    assert decision.evidence["rule_versions"] == (1,)
    starved = chamber_pair_decision(golden, candidate, history, 5, 3)
    assert not starved.allowed
    assert starved.reasons == ("insufficient_pairs:2<3",)
    assert starved.evidence["verdict"] == INSUFFICIENT_EVIDENCE

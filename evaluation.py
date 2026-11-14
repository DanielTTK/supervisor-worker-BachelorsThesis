"""Deterministic grading and case-clustered summaries for saved benchmark events."""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path


CLAIM_COLUMNS = (
    "backend", "case_id", "case_kind", "group_id", "arm", "condition", "scenario", "bundle_id", "stage",
    "field", "record_type", "role", "claim_id", "parent_claim_id", "origin",
    "first_observed_role", "introduced_in_call", "value", "source", "expected_value", "allowed_sources",
    "label", "valid_citation", "verdict", "forwarded", "caught", "escaped",
)
SUMMARY_COLUMNS = (
    "backend", "arm", "condition", "scenario", "case_kind", "scope", "metric", "numerator",
    "denominator", "percent", "ci_low", "ci_high", "clusters", "comparator",
    "comparator_numerator", "comparator_denominator",
)
BOOTSTRAP_METRICS = {
    "worker_hallucination_rate", "worker_originated_hallucination_rate",
    "supervisor_originated_hallucination_rate", "review_supervisor_originated_hallucination_rate",
    "synthesis_supervisor_originated_hallucination_rate", "review_miss_rate",
    "review_catch_recall", "review_false_alarm_rate", "review_escape_rate",
    "final_hallucination_rate", "final_case_error_rate", "final_exact_accuracy",
    "final_answer_coverage",
}
REVIEW_COMPARISONS = (
    ("reviewed", "weak_replay", None),
    ("injected_strong", "injected_weak", None),
    ("custom", "generic", "replay"),
)
FINAL_COMPARISONS = (
    ("generic", "workers_only"),
    ("custom", "generic"),
    ("custom", "workers_only"),
)


def _canonical(value):
    return " ".join(value.casefold().split()) if isinstance(value, str) else None


def _label(slot, target):
    value = _canonical(slot.get("value"))
    if value is None:
        return "abstention", None
    expected = _canonical(target["value"])
    if expected is None:
        label = "unsupported"
    elif value != expected:
        label = "contradicted"
    else:
        label = "supported"
    return label, label == "supported" and slot.get("source") in target["sources"]


def _wrong(label):
    return label in ("unsupported", "contradicted")


def _increment(buckets, key, numerator, denominator):
    counts = buckets.setdefault(key, [0, 0])
    counts[0] += numerator
    counts[1] += denominator


def _interval(counts, samples, paired=False):
    """Resample whole clusters, preserving the paired conditions within each cluster."""
    if samples < 1 or len(counts) < 2:
        return None, None
    rng = random.Random(314159)
    estimates = []
    for _ in range(samples):
        draw = rng.choices(counts, k=len(counts))
        denominator = sum(item[1] for item in draw)
        if denominator:
            numerator = sum(item[0] for item in draw)
            if paired:
                other_denominator = sum(item[3] for item in draw)
                if not other_denominator:
                    continue
                numerator_ratio = numerator / denominator
                other_ratio = sum(item[2] for item in draw) / other_denominator
                estimates.append(100 * (numerator_ratio - other_ratio))
            else:
                estimates.append(100 * numerator / denominator)
    if not estimates or min(estimates) == max(estimates):
        return None, None
    estimates.sort()
    return (
        round(estimates[int(0.025 * (len(estimates) - 1))], 4),
        round(estimates[int(0.975 * (len(estimates) - 1))], 4),
    )


def _validate_replay_pair(event, comparator):
    for item in (event, comparator):
        messages = item.get("messages")
        if (not isinstance(messages, list) or len(messages) < 2
                or any(not isinstance(message, dict) for message in messages)
                or messages[0].get("role") != "system"
                or any(message.get("role") == "system" for message in messages[1:])
                or not item.get("model") or not item.get("schema")
                or not item.get("options") or item.get("seed") is None):
            raise ValueError("Paired replay requires the original request metadata")
    if (event["messages"][0] == comparator["messages"][0]
            or event["messages"][1:] != comparator["messages"][1:]
            or any(event[key] != comparator[key] for key in ("schema", "model", "seed", "options"))):
        raise ValueError("Paired replay requests must differ only in the system prompt")


class _Grader:
    def __init__(self, cases):
        self.cases = {case["id"]: case for case in cases}
        self.claims = []
        self.metrics = defaultdict(dict)
        self.review_pairs = defaultdict(dict)
        self.final_pairs = defaultdict(dict)
        self.first_observed = {}

    def count(self, event, scope, metric, numerator, denominator):
        case = self.cases[event["case_id"]]
        group = case.get("group_id") or case["id"]
        for case_kind in ("all", case["kind"]):
            key = (event["arm"], event.get("scenario", "natural"), case_kind, scope, metric)
            _increment(self.metrics[key], group, numerator, denominator)

    def record_claim(self, event, field, slot, record_type, role, *, prior=None,
                     origin=None, claim_id=None, verdict=None, forwarded=None,
                     caught=None, escaped=None):
        case = self.cases[event["case_id"]]
        target = case["gold"][field]
        label, valid_citation = _label(slot, target)
        value = _canonical(slot.get("value"))
        parent_id = prior.get("claim_id") if prior and record_type == "review_correction" else None
        introduced = value is not None
        if introduced and prior and _canonical(prior.get("value")) == value:
            parent_id = prior.get("claim_id")
            origin = prior.get("origin") or "unknown"
            introduced = False
        scenario = event.get("scenario", "natural")
        first_role = None
        if value is not None:
            key = (case["id"], event["arm"], scenario, field, value)
            first_role = self.first_observed.setdefault(key, origin or role)
        self.claims.append({
            "case_id": case["id"], "case_kind": case["kind"],
            "group_id": case.get("group_id") or case["id"],
            "arm": event["arm"], "condition": event.get("condition", event["arm"]), "scenario": scenario,
            "bundle_id": event.get("bundle_id"), "stage": event["stage"],
            "field": field, "record_type": record_type, "role": role,
            "claim_id": claim_id, "parent_claim_id": parent_id,
            "origin": (origin or role) if value is not None else None,
            "first_observed_role": first_role,
            "introduced_in_call": introduced if value is not None and record_type != "review_candidate" else None,
            "value": slot.get("value"),
            "source": slot.get("source"), "expected_value": target["value"],
            "allowed_sources": json.dumps(target["sources"], ensure_ascii=False),
            "label": label, "valid_citation": valid_citation,
            "verdict": verdict, "forwarded": forwarded, "caught": caught,
            "escaped": escaped,
        })
        return label, valid_citation, introduced

    def worker(self, event):
        field = event["field"]
        slot = event["output"]
        label, valid_citation, introduced = self.record_claim(
            event, field, slot, "worker_output", "worker",
            prior=event.get("context", {}).get(field), claim_id=event.get("claim_id"),
        )
        asserted = label != "abstention"
        for scope in ("all", field):
            self.count(event, scope, "worker_hallucination_rate", _wrong(label), asserted)
            self.count(event, scope, "worker_originated_hallucination_rate",
                       introduced and _wrong(label), introduced)
            self.count(event, scope, "worker_unsupported_rate", label == "unsupported", asserted)
            self.count(event, scope, "worker_contradicted_rate", label == "contradicted", asserted)
            self.count(event, scope, "worker_citation_error_rate", valid_citation is False, asserted)
            self.count(event, scope, "worker_abstention_rate", not asserted, 1)

    def review(self, event, case):
        field = event["field"]
        candidate = event.get("candidate")
        if candidate is None:
            return
        target = case["gold"][field]
        output = (event.get("output") or {}) if event["status"] == "ok" else {}
        verdict = output.get("verdict")
        forwarded = event.get("forwarded")
        label, valid_citation = _label(candidate, target)
        wrong = _wrong(label)
        flagged = verdict in ("reject", "correct")
        forward_label, forward_citation = _label(forwarded or {}, target)
        escaped = wrong and _wrong(forward_label) and (
            _canonical(forwarded.get("value")) == _canonical(candidate.get("value"))
        )
        repaired = wrong and verdict == "correct" and (
            forward_label == "supported" and forward_citation is True
        )
        grounded = label == "supported" and valid_citation is True
        citation_only_error = label == "supported" and valid_citation is False
        candidate_origin = event.get("candidate_origin") or "worker"
        self.record_claim(
            event, field, candidate, "review_candidate", candidate_origin,
            origin=candidate_origin, claim_id=event.get("candidate_claim_id"),
            verdict=verdict, forwarded=forwarded is not None,
            caught=flagged if wrong else None, escaped=escaped if wrong else None,
        )
        correction = output if verdict == "correct" else {}
        corrected_label, corrected_citation, introduced = "abstention", None, False
        if verdict == "correct":
            prior = dict(candidate, origin=candidate_origin,
                         claim_id=event.get("candidate_claim_id"))
            corrected_label, corrected_citation, introduced = self.record_claim(
                event, field, correction, "review_correction", "supervisor",
                prior=prior, claim_id=event.get("claim_id"), verdict=verdict,
            )
        counts = {
            "review_catch_recall": (wrong and flagged, wrong),
            "review_miss_rate": (wrong and not flagged, wrong),
            "review_escape_rate": (escaped, wrong),
            "review_false_alarm_rate": (grounded and flagged, grounded),
            "review_verified_repair_rate": (repaired, wrong),
            "review_uncertain_rate": (verdict == "uncertain", 1),
            "review_missing_verdict_rate": (verdict is None, 1),
            "review_erroneous_acceptance_rate": (wrong and verdict == "accept", wrong),
            "review_citation_error_detection_rate": (citation_only_error and flagged,
                                                      citation_only_error),
            "review_supervisor_originated_hallucination_rate": (
                introduced and _wrong(corrected_label), introduced
            ),
            "supervisor_originated_hallucination_rate": (
                introduced and _wrong(corrected_label), introduced
            ),
            "review_correction_citation_error_rate": (
                corrected_citation is False, corrected_label != "abstention"
            ),
        }
        for scope in ("all", field):
            for metric, (numerator, denominator) in counts.items():
                self.count(event, scope, metric, numerator, denominator)
        scenario = event.get("scenario", "natural")
        arm = event["arm"]
        for left, right, required_scenario in REVIEW_COMPARISONS:
            if arm in (left, right) and required_scenario in (None, scenario):
                bundle = event.get("bundle_id")
                if required_scenario == "replay" and not bundle:
                    raise ValueError("Paired replay is missing a bundle ID")
                key = (case["id"], field, scenario, bundle)
                if required_scenario == "replay" and arm in self.review_pairs[key]:
                    raise ValueError(f"Duplicate {arm} replay for {bundle}")
                self.review_pairs[key][arm] = (event, wrong, flagged, grounded)

    def synthesis(self, event, case):
        output = (event.get("output") or {}) if event["status"] == "ok" else {}
        all_correct = event["status"] == "ok"
        wrong_count = 0
        answer_count = 0
        for field, target in case["gold"].items():
            slot = output.get(field)
            answerable = target["value"] is not None
            if slot is None:
                label, citation, introduced = None, None, False
            else:
                label, citation, introduced = self.record_claim(
                    event, field, slot, "final_output", "supervisor",
                    prior=event.get("context", {}).get(field),
                    claim_id=f"{event['event_id']}:{field}" if event.get("event_id") else None,
                )
            asserted = label is not None and label != "abstention"
            correct = (label == "supported" and citation is True) if answerable else (
                label == "abstention"
            )
            all_correct = all_correct and correct
            wrong_count += _wrong(label)
            answer_count += asserted
            for scope in ("all", field):
                self.count(event, scope, "final_hallucination_rate", _wrong(label), asserted)
                self.count(event, scope, "synthesis_supervisor_originated_hallucination_rate",
                           introduced and _wrong(label), introduced)
                self.count(event, scope, "supervisor_originated_hallucination_rate",
                           introduced and _wrong(label), introduced)
                self.count(event, scope, "synthesis_repeated_hallucination_rate",
                           not introduced and _wrong(label), not introduced and asserted)
                self.count(event, scope, "final_citation_error_rate", citation is False, asserted)
                self.count(event, scope, "final_answer_coverage", asserted and answerable, answerable)
                self.count(event, scope, "final_abstention_accuracy",
                           not answerable and label == "abstention", not answerable)
                self.count(event, scope, "final_grounded_slot_accuracy", correct, 1)
        self.count(event, "all", "final_exact_accuracy", all_correct, 1)
        self.count(event, "all", "final_case_error_rate", bool(wrong_count), 1)
        self.count(event, "all", "final_answered_case_error_rate", bool(wrong_count), bool(answer_count))
        if event.get("scenario", "natural") == "natural" and event["arm"] in (
            "workers_only", "generic", "custom"
        ):
            self.final_pairs[case["id"]][event["arm"]] = {
                "paired_final_hallucination_difference_pp": (wrong_count, answer_count),
                "paired_final_case_error_difference_pp": (bool(wrong_count), 1),
                "paired_final_exact_accuracy_difference_pp": (all_correct, 1),
            }

    def add_event(self, event):
        case = self.cases[event["case_id"]]
        stage = event["stage"]
        self.count(event, "all", f"{stage}_failure_rate", event["status"] != "ok", 1)
        if event.get("field"):
            self.count(event, event["field"], f"{stage}_failure_rate", event["status"] != "ok", 1)
        if stage == "review":
            self.review(event, case)
        elif stage == "synthesis":
            self.synthesis(event, case)
        elif stage == "worker" and event["status"] == "ok":
            self.worker(event)

    def paired_rows(self, samples):
        rows = []
        for left, right, required_scenario in REVIEW_COMPARISONS:
            groups = defaultdict(dict)
            for (case_id, field, scenario, _), pair in self.review_pairs.items():
                if required_scenario not in (None, scenario) or not ({left, right} & pair.keys()):
                    continue
                if left not in pair or right not in pair:
                    if required_scenario == "replay":
                        raise ValueError(f"Missing paired replay for {case_id} ({field})")
                    continue
                event, wrong, flagged, grounded = pair[left]
                comparator, _, other_flagged, _ = pair[right]
                if (event["candidate"] != comparator["candidate"]
                        or (event.get("context") or {}) != (comparator.get("context") or {})):
                    raise ValueError(f"Unmatched replay input for {case_id} ({field})")
                if required_scenario == "replay":
                    _validate_replay_pair(event, comparator)
                case = self.cases[case_id]
                group = case.get("group_id") or case_id
                for case_kind in ("all", case["kind"]):
                    for scope in ("all", field):
                        for metric, eligible, outcome, other_outcome in (
                            ("paired_miss_difference_pp", wrong, not flagged, not other_flagged),
                            ("paired_false_alarm_difference_pp", grounded, flagged, other_flagged),
                        ):
                            key = (scenario, case_kind, scope, metric)
                            counts = groups[key].setdefault(group, [0, 0, 0, 0])
                            counts[0] += eligible and outcome
                            counts[1] += eligible
                            counts[2] += eligible and other_outcome
                            counts[3] += eligible
            rows.extend(_comparison_rows(
                left, right, groups, samples, difference_numerator=required_scenario is None
            ))
        for left, right in FINAL_COMPARISONS:
            groups = defaultdict(dict)
            for case_id, pair in self.final_pairs.items():
                if left not in pair or right not in pair:
                    continue
                case = self.cases[case_id]
                group = case.get("group_id") or case_id
                for case_kind in ("all", case["kind"]):
                    for metric, (numerator, denominator) in pair[left].items():
                        other_num, other_den = pair[right][metric]
                        counts = groups[("natural", case_kind, "all", metric)].setdefault(
                            group, [0, 0, 0, 0]
                        )
                        counts[0] += numerator
                        counts[1] += denominator
                        counts[2] += other_num
                        counts[3] += other_den
            rows.extend(_comparison_rows(left, right, groups, samples))
        return rows

    def summary(self, samples):
        rows = []
        for (arm, scenario, case_kind, scope, metric), groups in sorted(self.metrics.items()):
            counts = list(groups.values())
            numerator, denominator = map(sum, zip(*counts))
            ci_low, ci_high = None, None
            if denominator and metric in BOOTSTRAP_METRICS:
                ci_low, ci_high = _interval(counts, samples)
            rows.append(_summary_row(
                arm, scenario, case_kind, scope, metric, numerator, denominator,
                ci_low, ci_high, len(counts)
            ))
        rows.extend(self.paired_rows(samples))
        return rows


def _comparison_rows(left, right, groups, samples, *, difference_numerator=False):
    for (scenario, case_kind, scope, metric), groups_by_id in sorted(groups.items()):
        counts = list(groups_by_id.values())
        numerator, denominator, other_num, other_den = map(sum, zip(*counts))
        ci_low, ci_high = _interval(counts, samples, paired=True)
        row = _summary_row(
            left, scenario, case_kind, scope, metric, numerator, denominator,
            ci_low, ci_high, len(counts), right, other_num, other_den,
        )
        row["percent"] = round(
            100 * (numerator / denominator - other_num / other_den), 4
        ) if denominator and other_den else None
        if difference_numerator:
            row["numerator"] = numerator - other_num
        yield row


def _summary_row(arm, scenario, case_kind, scope, metric, numerator, denominator, ci_low, ci_high,
                 clusters, comparator=None, comparator_numerator=None, comparator_denominator=None):
    return {
        "arm": arm, "condition": arm, "scenario": scenario, "case_kind": case_kind,
        "scope": scope, "metric": metric,
        "numerator": numerator, "denominator": denominator,
        "percent": round(100 * numerator / denominator, 4) if denominator else None,
        "ci_low": ci_low, "ci_high": ci_high, "clusters": clusters,
        "comparator": comparator, "comparator_numerator": comparator_numerator,
        "comparator_denominator": comparator_denominator,
    }


def _write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def grade(cases, events, out_dir, *, bootstrap_samples=2000, backend=None):
    """Score model outputs against held-out gold and write claims and summary data."""
    grader = _Grader(cases)
    for event in events:
        grader.add_event(event)
    rows = grader.summary(bootstrap_samples)
    for row in (*rows, *grader.claims):
        row["backend"] = backend
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "claims.csv", CLAIM_COLUMNS, grader.claims)
    _write_csv(output / "summary.csv", SUMMARY_COLUMNS, rows)
    (output / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return rows

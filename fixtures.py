"""Deterministic, source-grounded cases for the local supervisor benchmark."""

import hashlib
import random


FIELDS = ("revision", "setting", "exception")
CASE_KINDS = ("answerable", "missing-answer", "false-premise", "conflicting-record")
STAGE_TASKS = {
    "revision": (
        "Find the currently assigned revision for the requested unit. Follow the "
        "record authority policy and copy the revision code exactly. A question's "
        "suggested revision is a claim to check against the records."
    ),
    "setting": (
        "Find the SETTING code for the requested unit and its reviewed active "
        "revision. Use the reviewed revision as the lookup key. Copy the code "
        "exactly. Abstain if the revision is unknown or no authoritative row matches."
    ),
    "exception": (
        "Find the EXCEPTION code for the requested unit, reviewed active revision, "
        "and reviewed setting. Use both reviewed values as lookup keys. Copy the "
        "code exactly. Abstain if either key is unknown or no authoritative row matches."
    ),
}

_EXCEPTION_CODES = (
    "AFTER_CALIBRATION",
    "DURING_INSPECTION",
    "OVERNIGHT_ONLY",
    "SENSOR_OFFLINE",
    "MAINTENANCE_MODE",
    "BACKUP_ACTIVE",
    "MANUAL_OVERRIDE",
    "DURING_TEST_CYCLE",
)
_AUTHORITY_POLICY = (
    "For current revision assignments, setting codes, and exception codes, use "
    "APPROVED records. PROVISIONAL and ARCHIVED records cannot establish current "
    "values. Match the requested unit exactly. Setting codes depend on unit and "
    "active revision. Exception codes depend on unit, active revision, and setting. "
    "If a required approved record or upstream value is absent, report insufficient evidence."
)


def generate_cases(count: int, seed: int) -> list[dict]:
    """Return balanced dossiers, with the answer key separate from visible evidence."""
    if count < 1:
        raise ValueError("count must be at least 1")

    rng = random.Random(seed)
    cases = []
    missing_index = 0
    while len(cases) < count:
        kinds = list(CASE_KINDS)
        rng.shuffle(kinds)
        for kind in kinds[: count - len(cases)]:
            missing = None
            if kind == "missing-answer":
                missing = FIELDS[missing_index % len(FIELDS)]
                missing_index += 1
            cases.append(_make_case(rng, seed, len(cases) + 1, kind, missing))
    return cases


def _make_case(rng: random.Random, seed: int, index: int, kind: str, missing: str | None) -> dict:
    dossier = hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()[:10]
    case_id = f"dossier-{dossier}"
    unit = f"UNIT-{dossier[:6].upper()}"
    nearby_unit = f"{unit}-B"
    active, other = (f"R-{number:02d}" for number in rng.sample(range(2, 80), 2))
    setting, other_setting, draft_setting = (
        f"S-{number:03d}" for number in rng.sample(range(10, 900), 3)
    )
    exception, other_exception, draft_exception = rng.sample(_EXCEPTION_CODES, 3)

    # Stable span IDs are local to each dossier. Shuffle the evidence order so
    # authority and matching keys, rather than list position, decide each answer.
    records = {
        "policy": _AUTHORITY_POLICY,
        "archive_assignment": f"ARCHIVED assignment | Unit {unit} | Revision {other}",
        "other_setting": f"APPROVED setting | Unit {unit} | Revision {other} | SETTING {other_setting}",
        "other_exception": (
            f"APPROVED exception | Unit {unit} | Revision {other} | "
            f"SETTING {other_setting} | EXCEPTION {other_exception}"
        ),
        "nearby_setting": (
            f"APPROVED setting | Unit {nearby_unit} | Revision {active} | SETTING {draft_setting}"
        ),
    }
    if missing != "revision":
        records["assignment"] = f"APPROVED assignment | Unit {unit} | Revision {active}"
    if missing != "setting":
        records["setting"] = f"APPROVED setting | Unit {unit} | Revision {active} | SETTING {setting}"
    if missing != "exception":
        records["exception"] = (
            f"APPROVED exception | Unit {unit} | Revision {active} | "
            f"SETTING {setting} | EXCEPTION {exception}"
        )
    if kind == "conflicting-record":
        records["draft_assignment"] = f"PROVISIONAL assignment | Unit {unit} | Revision {other}"
        records["draft_setting"] = (
            f"PROVISIONAL setting | Unit {unit} | Revision {active} | SETTING {draft_setting}"
        )
        records["draft_exception"] = (
            f"PROVISIONAL exception | Unit {unit} | Revision {active} | "
            f"SETTING {setting} | EXCEPTION {draft_exception}"
        )

    question = f"Which revision is active for {unit}, and what are its setting and exception codes?"
    if kind == "false-premise":
        question = (
            f"I was told {other} is active for {unit}. Verify its actual active "
            "revision, setting code, and exception code from the dossier."
        )

    # A missing prerequisite makes downstream answers unidentifiable even if
    # the dossier happens to contain a plausible setting or exception record.
    revision_known = missing != "revision"
    setting_known = revision_known and missing != "setting"
    exception_known = setting_known and missing != "exception"
    span_keys = list(records)
    rng.shuffle(span_keys)
    span_ids = {key: f"D{position:02d}" for position, key in enumerate(span_keys, 1)}
    gold = {
        "revision": _expected(active, span_ids["assignment"] if revision_known else None),
        "setting": _expected(setting, span_ids["setting"] if setting_known else None),
        "exception": _expected(exception, span_ids["exception"] if exception_known else None),
    }
    return {
        "id": case_id,
        "kind": kind,
        "question": question,
        "evidence": {span_id: records[key] for key, span_id in span_ids.items()},
        "gold": gold,
    }


def _expected(value: str, source: str | None) -> dict:
    if source is None:
        return {"value": None, "sources": []}
    return {"value": value, "sources": [source]}

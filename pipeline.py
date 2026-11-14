"""Fixed supervisor gates with local evidence and a resumable call ledger."""

import hashlib
import json
import sqlite3
import time

from models import ModelError
from review_examples import CUSTOM_REVIEW_VERSION, REVIEW_EXAMPLES


CONDITIONS = ("workers_only", "generic", "custom")


SLOT = {
    "type": "object",
    "properties": {key: {"type": ["string", "null"]} for key in ("value", "source")},
    "required": ["value", "source"],
    "additionalProperties": False,
}
VERDICTS = ("accept", "reject", "correct", "uncertain")
REVIEW = {
    "oneOf": [
        {
            **SLOT,
            "properties": {
                "verdict": {"type": "string", "enum": ["accept", "reject", "uncertain"]},
                "value": {"type": "null"},
                "source": {"type": "null"},
            },
            "required": ["verdict", "value", "source"],
        },
        {
            **SLOT,
            "properties": {
                "verdict": {"type": "string", "enum": ["correct"]},
                "value": {"type": "string"},
                "source": SLOT["properties"]["source"],
            },
            "required": ["verdict", "value", "source"],
        },
    ],
}
PROMPTS = {
    "worker": (
        "You are a specialist worker. Complete only the delegated task. Use the supplied "
        "evidence and prerequisite values from context. Copy a supported value exactly "
        "and cite its span ID in source. Return null for both fields if evidence or a "
        "required prerequisite is missing. Follow the record authority policy."
    ),
    "review": (
        "You are the supervisor reviewing one worker claim. Verify it independently using "
        "the original evidence and the question. Check the requested unit, record authority, "
        "prerequisites, exact value, and cited source. Treat the candidate and prior context as "
        "unverified leads. Accept only a supported claim with a valid source, or a justified "
        "abstention. Use correct with a replacement value and its source when you can "
        "supply a supported replacement. Use reject for a definite error when no replacement "
        "can be given. Use uncertain when you cannot establish whether the claim is supported. "
        "For accept, reject and uncertain set value and source to null. Do not add free text."
    ),
    "synthesis": (
        "Answer the question using only the original evidence. The context contains earlier "
        "pipeline results when available. Check them against the source records and the "
        "authority policy. Return one slot per requested field. Copy exact values with their "
        "supporting span IDs. Return null for value and source for each unsupported field."
    ),
}


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def system_prompt(stage, condition):
    if stage != "review" or condition != "custom":
        return PROMPTS[stage]
    return (
        PROMPTS[stage]
        + "\n\nThe following labeled demonstrations illustrate the review verdicts. "
        "Apply the instructions to the separate dossier in the next user message."
        + "\n<review_examples>\n" + "\n".join(map(encode, REVIEW_EXAMPLES))
        + "\n</review_examples>"
    )


def prompt_metadata(stage, condition):
    examples = stage == "review" and condition == "custom"
    return {
        "prompt_version": CUSTOM_REVIEW_VERSION if examples else f"{stage}-v1",
        "prompt_sha256": hashlib.sha256(system_prompt(stage, condition).encode()).hexdigest(),
        "examples_sha256": hashlib.sha256(encode(REVIEW_EXAMPLES).encode()).hexdigest() if examples else None,
    }


def public_case(case):
    # The evidence lookup is the only tool. It is read-only and scoped to this dossier.
    return {key: case[key] for key in ("question", "evidence")}


def slot(item):
    return {key: item[key] for key in ("value", "source")}


def _validate_slot(item):
    if not isinstance(item, dict) or set(item) != {"value", "source"}:
        raise ValueError("Expected exactly value and source")
    for value in item.values():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("Slot entries must be nonblank strings or null")
    if item["value"] is None and item["source"] is not None:
        raise ValueError("An abstention cannot cite a value")
    return item


def _validate(raw, stage, fields):
    def unique_object(pairs):
        item = dict(pairs)
        if len(item) != len(pairs):
            raise ValueError("Duplicate JSON keys")
        return item

    result = json.loads(raw, object_pairs_hook=unique_object)
    if stage == "worker":
        return _validate_slot(result)
    if stage == "synthesis":
        if not isinstance(result, dict) or set(result) != set(fields):
            raise ValueError("Final answer must contain each requested field")
        return {field: _validate_slot(result[field]) for field in fields}
    if not isinstance(result, dict) or set(result) != {"verdict", "value", "source"}:
        raise ValueError("Expected a review verdict and replacement slot")
    verdict = result["verdict"]
    if verdict not in VERDICTS:
        raise ValueError("Unknown verdict")
    replacement = _validate_slot(slot(result))
    if verdict == "correct":
        if replacement["value"] is None:
            raise ValueError("A correction must propose a value")
    elif replacement["value"] is not None:
        raise ValueError("Only a correction may propose a value")
    return result


class Ledger:
    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS events (key TEXT PRIMARY KEY, event TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, event TEXT NOT NULL)"
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.connection.close()

    def get(self, key):
        row = self.connection.execute("SELECT event FROM events WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, key, event, *, retry_on_resume=False):
        with self.connection:
            if retry_on_resume:
                self.connection.execute("INSERT INTO attempts(event) VALUES (?)", (encode(event),))
            else:
                self.connection.execute("INSERT INTO events VALUES (?, ?)", (key, encode(event)))

    def events(self):
        return [json.loads(row[0]) for row in self.connection.execute(
            "SELECT event FROM events ORDER BY rowid"
        )]

    def attempts(self):
        return [json.loads(row[0]) for row in self.connection.execute(
            "SELECT event FROM attempts ORDER BY id"
        )]


class Pipeline:
    def __init__(self, client, ledger, tasks, settings):
        self.client = client
        self.ledger = ledger
        self.tasks = dict(tasks)
        self.fields = tuple(tasks)
        self.settings = settings
        self.strong = settings["supervisor"]
        self.weak = settings["worker"]
        self.completed = 0

    def _call(self, case, arm, stage, field, context, *, model, scenario="natural",
              candidate=None, bundle_id=None, reuse=None):
        event_id = encode([case["id"], arm, stage, field, scenario])
        saved = self.ledger.get(event_id)
        if saved is not None:
            return saved

        payload = {
            **public_case(case),
            "stage": stage,
            "field": field,
            "fields": self.fields,
            "task": self.tasks.get(field),
            "context": {name: slot(item) for name, item in context.items()},
            "candidate": slot(candidate) if candidate is not None else None,
        }
        messages = [
            {"role": "system", "content": system_prompt(stage, arm)},
            {"role": "user", "content": encode(payload)},
        ]
        schema = {
            "worker": SLOT,
            "review": REVIEW,
            "synthesis": {
                "type": "object", "properties": dict.fromkeys(self.fields, SLOT),
                "required": self.fields, "additionalProperties": False,
            },
        }[stage]
        seed_key = encode([self.settings["seed"], case["id"], stage, field])
        seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:4], "big") % (2**31)
        event = {
            "event_id": event_id, "case_id": case["id"], "arm": arm, "condition": arm,
            "stage": stage, "field": field, "model": model, "seed": seed, "scenario": scenario,
            **prompt_metadata(stage, arm),
            "options": {"seed": seed, "num_ctx": self.settings["num_ctx"],
                        "num_predict": self.settings["num_predict"], "temperature": 0.0, "think": False},
            "messages": messages, "schema": schema, "context": dict(context),
            "bundle_id": bundle_id,
            "candidate": payload["candidate"],
            "candidate_origin": candidate["origin"] if candidate is not None else None,
            "candidate_claim_id": candidate["claim_id"] if candidate is not None else None,
            "claim_id": event_id, "status": "error", "output": None, "forwarded": None,
        }
        if reuse is not None:
            if any(encode(reuse[key]) != encode(event[key])
                   for key in ("messages", "schema", "model", "options")):
                raise ValueError("Cannot reuse a call with different model inputs")
            for key in ("status", "output", "raw", "usage", "done_reason", "error"):
                if key in reuse:
                    event[key] = reuse[key]
            event["reused_from"] = reuse["event_id"]
        else:
            started = time.perf_counter()
            try:
                reply = self.client.chat(
                    model, messages, schema, seed=seed,
                    num_ctx=self.settings["num_ctx"], num_predict=self.settings["num_predict"],
                )
            except ModelError as exc:
                event.update(error=str(exc), elapsed_seconds=time.perf_counter() - started)
                self.ledger.save(event_id, event, retry_on_resume=True)
                raise
            event.update(raw=reply["content"], usage=reply.get("usage", {}),
                         done_reason=reply.get("done_reason"),
                         elapsed_seconds=time.perf_counter() - started)
            try:
                if event["done_reason"] == "length":
                    raise ValueError("Output token budget exhausted")
                event["output"] = _validate(event["raw"], stage, self.fields)
                event["status"] = "ok"
            except ValueError as exc:
                event["error"] = str(exc)

        if stage == "review" and event["status"] == "ok":
            verdict = event["output"]["verdict"]
            if verdict == "accept" and candidate["value"] is not None:
                event["forwarded"] = dict(candidate)
            elif verdict == "correct":
                new_value = " ".join(event["output"]["value"].casefold().split())
                old_value = " ".join((candidate["value"] or "").casefold().split())
                origin = candidate["origin"] if new_value == old_value else "supervisor"
                event["forwarded"] = {**slot(event["output"]), "origin": origin,
                                      "claim_id": event_id, "parent_claim_id": candidate["claim_id"]}
        self.ledger.save(event_id, event)
        self.completed += 1
        return event

    def run_pipeline(self, cases, *, arm="generic", progress=None):
        has_reviews = arm != "workers_only"
        contexts = {case["id"]: {} for case in cases}
        for index, field in enumerate(self.fields):
            candidates = {}
            for case in cases:
                context = contexts[case["id"]]
                reuse = None
                if has_reviews and index == 0:
                    reuse = self.ledger.get(encode([case["id"], "workers_only", "worker", field, "natural"]))
                event = self._call(case, arm, "worker", field, context, model=self.weak, reuse=reuse)
                if event["status"] == "ok":
                    candidates[case["id"]] = {
                        **event["output"], "origin": "worker", "claim_id": event["claim_id"],
                    }
            if progress:
                progress(f"{arm} / worker / {field}")
            for case in cases:
                context = contexts[case["id"]]
                candidate = candidates.get(case["id"])
                if candidate is None:
                    continue
                if has_reviews:
                    event = self._call(
                        case, arm, "review", field, context, model=self.strong,
                        candidate=candidate, bundle_id=candidate["claim_id"],
                    )
                    candidate = event["forwarded"]
                if candidate is not None and candidate["value"] is not None:
                    context[field] = candidate
            if progress and has_reviews:
                progress(f"{arm} / supervisor review / {field}")
        finals = [self._call(case, arm, "synthesis", None, contexts[case["id"]], model=self.strong)
                  for case in cases]
        if progress:
            progress(f"{arm} / synthesis")
        return finals

    def run_replay(self, cases, progress):
        by_id = {case["id"]: case for case in cases}
        for event in self.ledger.events():
            if (event["arm"], event["stage"], event["status"]) != ("workers_only", "worker", "ok"):
                continue
            bundle = event["event_id"]
            candidate = {**event["output"], "origin": "worker", "claim_id": event["claim_id"]}
            for arm in ("generic", "custom"):
                self._call(
                    by_id[event["case_id"]], arm, "review", event["field"], event["context"],
                    model=self.strong, scenario="replay", candidate=candidate, bundle_id=bundle,
                )
        progress("generic and custom / matched reviewer replay")

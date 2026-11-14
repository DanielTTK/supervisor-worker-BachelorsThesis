"""Integration checks for the three-condition, offline thesis comparison."""

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pipeline import encode
from review_examples import REVIEW_EXAMPLES


ROOT = Path(__file__).resolve().parent
ARMS = ("workers_only", "generic", "custom")
FIELDS = ("revision", "setting", "exception")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def slot(claim):
    return {key: claim[key] for key in ("value", "source")}


def user_payload(event):
    messages = [message for message in event["messages"] if message["role"] == "user"]
    return json.loads(messages[-1]["content"])


class ComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use the writable workspace because the shell's default temp root is sandboxed.
        directory = tempfile.TemporaryDirectory(prefix="comparison-test-", dir=ROOT)
        cls.addClassCleanup(directory.cleanup)
        cls.out = Path(directory.name) / "run"
        cls.run_args = (
            "run", "--fake", "--out", str(cls.out), "--cases", "4",
            "--seed", "101", "--bootstrap", "20",
        )
        cls.run_cli(*cls.run_args)
        cls.events = read_jsonl(cls.out / "events.jsonl")
        cls.cases = {case["id"]: case for case in read_json(cls.out / "cases.json")}
        cls.index = {
            (event["case_id"], event["arm"], event["scenario"], event["stage"], event["field"]): event
            for event in cls.events
        }

    @classmethod
    def run_cli(cls, *args):
        result = subprocess.run(
            [sys.executable, "-B", "-X", "utf8", str(ROOT / "thesis.py"), *args],
            cwd=ROOT,
            capture_output=True,
            encoding="utf-8",
            timeout=90,
            check=False,
        )
        if result.returncode:
            raise AssertionError(f"CLI failed with {result.returncode}\n{result.stdout}\n{result.stderr}")
        return result.stdout

    def event(self, case_id, arm, stage, field=None, *, scenario="natural"):
        return self.index[case_id, arm, scenario, stage, field]

    def assert_same_request_except_context(self, baseline, other):
        for key in ("stage", "field", "model", "seed", "schema", "options", "prompt_sha256"):
            self.assertEqual(baseline[key], other[key], key)
        self.assertEqual(
            [message for message in baseline["messages"] if message["role"] == "system"],
            [message for message in other["messages"] if message["role"] == "system"],
        )
        baseline_payload = user_payload(baseline)
        other_payload = user_payload(other)
        baseline_payload.pop("context")
        other_payload.pop("context")
        self.assertEqual(baseline_payload, other_payload)

    def test_natural_workflows_share_tasks_and_forward_the_expected_claims(self):
        self.assertEqual(len(self.cases), 4)
        self.assertEqual(len(self.events), len(self.index))
        self.assertEqual({event["arm"] for event in self.events}, set(ARMS))
        self.assertEqual({event["scenario"] for event in self.events}, {"natural", "replay"})
        self.assertFalse(any(
            event["stage"] == "review" and event["arm"] == "workers_only"
            for event in self.events
        ))

        changed_contexts = 0
        for case_id, case in self.cases.items():
            for arm in ARMS:
                expected_context = {}
                for field in FIELDS:
                    with self.subTest(case=case_id, arm=arm, field=field):
                        worker = self.event(case_id, arm, "worker", field)
                        baseline = self.event(case_id, "workers_only", "worker", field)
                        self.assert_same_request_except_context(baseline, worker)
                        payload = user_payload(worker)
                        self.assertEqual(payload["context"], expected_context)
                        self.assertEqual(payload["evidence"], case["evidence"])
                        self.assertEqual(payload["question"], case["question"])
                        self.assertNotIn("gold", payload)
                        self.assertEqual(worker["status"], "ok")
                        if arm == "workers_only":
                            forwarded = worker["output"]
                        else:
                            review = self.event(case_id, arm, "review", field)
                            self.assertEqual(review["status"], "ok")
                            self.assertEqual(review["candidate"], worker["output"])
                            self.assertEqual(review["candidate_claim_id"], worker["claim_id"])
                            self.assertEqual(user_payload(review)["context"], expected_context)
                            forwarded = review["forwarded"]
                            if forwarded is None or slot(forwarded) != worker["output"]:
                                changed_contexts += 1
                        if forwarded and forwarded["value"] is not None:
                            expected_context[field] = slot(forwarded)
                final = self.event(case_id, arm, "synthesis")
                self.assert_same_request_except_context(self.event(case_id, "workers_only", "synthesis"), final)
                self.assertEqual(user_payload(final)["context"], expected_context)
                self.assertEqual(final["status"], "ok")
        self.assertGreater(changed_contexts, 0, "scripted review should change what flows downstream")

    def test_paired_replay_changes_only_the_review_prompt(self):
        for case_id in self.cases:
            for field in FIELDS:
                baseline = self.event(case_id, "workers_only", "worker", field)
                generic = self.event(case_id, "generic", "review", field, scenario="replay")
                custom = self.event(case_id, "custom", "review", field, scenario="replay")
                with self.subTest(case=case_id, field=field):
                    for replay in (generic, custom):
                        self.assertEqual(replay["candidate"], baseline["output"])
                        self.assertEqual(replay["context"], baseline["context"])
                        self.assertEqual(replay["candidate_claim_id"], baseline["claim_id"])
                        self.assertEqual(replay["status"], "ok")
                    for key in ("model", "options", "schema", "seed", "bundle_id", "candidate"):
                        self.assertEqual(generic[key], custom[key], key)
                    for replay in (generic, custom):
                        prompts = [message["content"] for message in replay["messages"] if message["role"] == "system"]
                        self.assertEqual(len(prompts), 1)
                        self.assertEqual(replay["prompt_sha256"], hashlib.sha256(prompts[0].encode()).hexdigest())
                    self.assertNotEqual(generic["prompt_sha256"], custom["prompt_sha256"])
                    self.assertEqual(
                        [message for message in generic["messages"] if message["role"] != "system"],
                        [message for message in custom["messages"] if message["role"] != "system"],
                    )

    def test_metadata_and_exports_distinguish_condition_from_replay(self):
        manifest = read_json(self.out / "manifest.json")
        self.assertEqual(manifest["backend"], "scripted_smoke_only")
        self.assertEqual(manifest["replay_source"], "workers_only")
        self.assertEqual(set(manifest["conditions"]), set(ARMS))
        self.assertNotIn("review", manifest["conditions"]["workers_only"])
        self.assertNotEqual(
            manifest["conditions"]["generic"]["review"], manifest["conditions"]["custom"]["review"],
        )
        for stage in ("worker", "synthesis"):
            self.assertEqual(manifest["conditions"]["workers_only"][stage], manifest["conditions"]["generic"][stage])
            self.assertEqual(manifest["conditions"]["generic"][stage], manifest["conditions"]["custom"][stage])
        example_hash = hashlib.sha256(encode(REVIEW_EXAMPLES).encode()).hexdigest()
        for event in self.events:
            with self.subTest(event=event["event_id"]):
                self.assertEqual(event["condition"], event["arm"])
                metadata = manifest["conditions"][event["condition"]][event["stage"]]
                for key in ("prompt_version", "prompt_sha256", "examples_sha256"):
                    self.assertEqual(event[key], metadata[key])
                custom_review = event["condition"] == "custom" and event["stage"] == "review"
                self.assertEqual(event["examples_sha256"], example_hash if custom_review else None)

        with (self.out / "summary.csv").open(encoding="utf-8", newline="") as stream:
            summary = list(csv.DictReader(stream))
        by_condition = {
            (row["arm"], row["scenario"], row["metric"])
            for row in summary
            if row["case_kind"] == row["scope"] == "all"
        }
        for arm in ARMS:
            self.assertIn((arm, "natural", "worker_hallucination_rate"), by_condition)
            self.assertIn((arm, "natural", "final_hallucination_rate"), by_condition)
            self.assertIn((arm, "natural", "final_exact_accuracy"), by_condition)
        for arm in ("generic", "custom"):
            self.assertIn((arm, "natural", "review_miss_rate"), by_condition)
            self.assertIn((arm, "replay", "review_miss_rate"), by_condition)
            self.assertNotIn((arm, "replay", "worker_hallucination_rate"), by_condition)
            self.assertNotIn((arm, "replay", "final_exact_accuracy"), by_condition)
        self.assertNotIn(("workers_only", "natural", "review_miss_rate"), by_condition)
        paired = {
            (row["condition"], row["comparator"], row["scenario"], row["metric"])
            for row in summary if row["case_kind"] == row["scope"] == "all" and row["comparator"]
        }
        self.assertIn(("custom", "generic", "replay", "paired_miss_difference_pp"), paired)
        for left, right in (("custom", "generic"), ("custom", "workers_only"), ("generic", "workers_only")):
            self.assertIn((left, right, "natural", "paired_final_hallucination_difference_pp"), paired)

    def test_resume_and_offline_rescore_preserve_events_and_results(self):
        summary = (self.out / "summary.csv").read_bytes()
        original_events = (self.out / "events.jsonl").read_bytes()
        self.run_cli(*self.run_args)
        self.assertEqual((self.out / "events.jsonl").read_bytes(), original_events)
        self.run_cli("score", "--out", str(self.out), "--bootstrap", "20")
        self.assertEqual((self.out / "summary.csv").read_bytes(), summary)

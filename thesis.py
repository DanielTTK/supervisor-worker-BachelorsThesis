"""Run, resume and score a local Supervisor–Worker experiment."""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

from evaluation import grade
from fixtures import STAGE_TASKS, generate_cases
from models import FakeClient, ModelError, OllamaClient
from pipeline import CONDITIONS, Ledger, Pipeline, prompt_metadata


WORKER = "qwen3:1.7b-q4_K_M"
SUPERVISOR = "qwen3:8b-q4_K_M"
IMPLEMENTATION = ("fixtures.py", "models.py", "pipeline.py", "review_examples.py", "thesis.py")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_text(path, text):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(text, encoding="utf-8")
    pending.replace(path)


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def settings(args):
    if args.worker == args.supervisor:
        raise ValueError("Choose different worker and supervisor model tags")
    return {key: getattr(args, key) for key in ("worker", "supervisor", "seed", "num_ctx", "num_predict")}


def model_client(args):
    if args.fake:
        return FakeClient(supervisor_model=args.supervisor)
    return OllamaClient(keep_alive=args.keep_alive, timeout=args.timeout)


def prepare(args, client, config):
    cases = generate_cases(args.cases, args.seed)
    models = client.metadata([args.worker, args.supervisor])
    code = Path(__file__).resolve().parent
    manifest = {
        "backend": "scripted_smoke_only" if args.fake else "ollama",
        "inference": config, "runtime": models, "python": sys.version,
        "platform": platform.platform(),
        "cases": args.cases, "replay_source": "workers_only",
        "conditions": {
            condition: {
                stage: prompt_metadata(stage, condition)
                for stage in ("worker", "review", "synthesis")
                if stage != "review" or condition != "workers_only"
            }
            for condition in CONDITIONS
        },
        "keep_alive": args.keep_alive, "timeout": args.timeout,
        "dataset_sha256": digest(canonical(cases).encode()),
        "code_sha256": {name: digest((code / name).read_bytes()) for name in IMPLEMENTATION},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_file = args.out / "manifest.json"
    if manifest_file.exists():
        if canonical(manifest) != canonical(read_json(manifest_file)):
            raise ValueError("Run configuration, fixtures, code or model metadata changed. Choose a new --out.")
        recorded_cases = read_json(args.out / "cases.json")
        if canonical(cases) != canonical(recorded_cases):
            raise ValueError("Saved fixture data differs from the experiment manifest")
    else:
        if (args.out / "runs.sqlite3").exists():
            raise ValueError("The run database already exists without its manifest. Choose a new --out.")
        write_json(args.out / "cases.json", cases)
        write_json(manifest_file, manifest)
    return cases


def export(out_dir, ledger):
    for filename, rows in (("events.jsonl", ledger.events()), ("attempts.jsonl", ledger.attempts())):
        write_text(out_dir / filename, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def score(out_dir, bootstrap):
    if not (out_dir / "complete.json").exists():
        raise ValueError("Run is incomplete. Resume it with the original run command before scoring.")
    manifest = read_json(out_dir / "manifest.json")
    cases = read_json(out_dir / "cases.json")
    if digest(canonical(cases).encode()) != manifest["dataset_sha256"]:
        raise ValueError("Saved fixture data differs from the experiment manifest")
    database = out_dir / "runs.sqlite3"
    if not database.is_file():
        raise ValueError(f"Missing run database: {database}")
    with Ledger(database) as ledger:
        events = ledger.events()
        rows = grade(cases, events, out_dir, bootstrap_samples=bootstrap, backend=manifest["backend"])
    write_json(out_dir / "scoring.json", {
        "backend": manifest["backend"], "bootstrap_samples": bootstrap,
        "scorer_sha256": digest(Path(__file__).with_name("evaluation.py").read_bytes()),
    })
    label = "Scripted smoke test" if manifest["backend"] == "scripted_smoke_only" else "Local Ollama"
    print(f"{label}. Scored {len(events)} events. Results: {out_dir.resolve()}", flush=True)
    headline = {
        "worker_hallucination_rate": "Worker hallucination rate",
        "review_miss_rate": "Supervisor miss rate",
        "review_escape_rate": "Worker error escape rate",
        "supervisor_originated_hallucination_rate": "Supervisor-originated hallucination rate",
        "final_exact_accuracy": "Exact final accuracy",
        "final_hallucination_rate": "Final hallucination rate",
    }
    for row in rows:
        if (row["scenario"], row["case_kind"], row["scope"]) != ("natural", "all", "all"):
            continue
        if row["metric"] not in headline:
            continue
        percent = "undefined" if row["percent"] is None else f"{row['percent']:.2f}%"
        print(f"  {row['arm']} / {headline[row['metric']]}  {percent} ({row['numerator']}/{row['denominator']})")


def run(args):
    config = settings(args)
    client = model_client(args)
    cases = prepare(args, client, config)
    if args.fake:
        print("Scripted smoke test. The output is not model evidence.", flush=True)
    with Ledger(args.out / "runs.sqlite3") as ledger:
        pipeline = Pipeline(client, ledger, STAGE_TASKS, config)

        def progress(phase):
            print(f"{phase} complete ({pipeline.completed} new events this session)", flush=True)

        try:
            for condition in CONDITIONS:
                pipeline.run_pipeline(cases, arm=condition, progress=progress)
            pipeline.run_replay(cases, progress)
        finally:
            export(args.out, ledger)
        write_json(args.out / "complete.json", {"events": len(ledger.events())})
    score(args.out, args.bootstrap)


def answer(args):
    case = read_json(args.input)
    if (not isinstance(case, dict) or not isinstance(case.get("question"), str)
            or not case["question"].strip() or not isinstance(case.get("evidence"), dict)
            or not case["evidence"] or any(not isinstance(value, str) for value in case["evidence"].values())):
        raise ValueError("Input must contain a question and an evidence object of span IDs to source text")
    config = settings(args)
    client = model_client(args)
    client.metadata([args.worker, args.supervisor])
    with Ledger(":memory:") as ledger:
        final = Pipeline(client, ledger, STAGE_TASKS, config).run_pipeline([{"id": "user", **case}])[0]
    if final["status"] != "ok":
        raise ValueError(f"Synthesis failed: {final['error']}")
    print(json.dumps(final["output"], ensure_ascii=False, indent=2))


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def add_runtime_options(parser):
    parser.add_argument("--worker", default=WORKER)
    parser.add_argument("--supervisor", default=SUPERVISOR)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-ctx", type=positive, default=4096)
    parser.add_argument("--num-predict", type=positive, default=256)
    parser.add_argument("--timeout", type=positive, default=300)
    parser.add_argument("--keep-alive", default="5m")
    parser.add_argument("--fake", action="store_true", help="use scripted smoke-test responses")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("run", help="run or resume, then score")
    add_runtime_options(benchmark)
    benchmark.add_argument("--out", type=Path, required=True)
    benchmark.add_argument("--cases", type=positive, default=24)
    benchmark.add_argument("--bootstrap", type=positive, default=2000)
    scoring = commands.add_parser("score", help="regenerate results without model calls")
    scoring.add_argument("--out", type=Path, required=True)
    scoring.add_argument("--bootstrap", type=positive, default=2000)
    interactive = commands.add_parser("answer", help="answer one user-supplied dossier")
    add_runtime_options(interactive)
    interactive.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "run":
            run(args)
        elif args.command == "score":
            score(args.out, args.bootstrap)
        else:
            answer(args)
    except (ModelError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Use the same run command to resume.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

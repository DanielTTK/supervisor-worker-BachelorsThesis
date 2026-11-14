"""Generate readable Markdown and LaTeX reports from benchmark run directories."""

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path


CONDITION_LABELS = {
    "workers_only": "Workers only",
    "generic": "Generic review",
    "custom": "Custom review",
}
METRIC_LABELS = {
    "final_exact_accuracy": "Exact final answers",
    "final_hallucination_rate": "Final hallucinations",
    "final_case_error_rate": "Final case errors",
    "final_answer_coverage": "Final answer coverage",
    "worker_hallucination_rate": "Worker hallucinations",
    "worker_failure_rate": "Worker failures",
    "review_miss_rate": "Reviewer misses",
    "review_escape_rate": "Worker errors escaping review",
    "review_verified_repair_rate": "Verified reviewer repairs",
    "supervisor_originated_hallucination_rate": "Supervisor-originated hallucinations",
    "synthesis_failure_rate": "Synthesis failures",
    "review_catch_recall": "Reviewer catch recall",
    "review_false_alarm_rate": "Reviewer false alarms",
    "review_supervisor_originated_hallucination_rate": "Reviewer-originated hallucinations",
    "review_failure_rate": "Reviewer failures",
    "paired_final_hallucination_difference_pp": "Final hallucinations (end-to-end)",
    "paired_final_case_error_difference_pp": "Final case errors (end-to-end)",
    "paired_final_exact_accuracy_difference_pp": "Exact final answers (end-to-end)",
    "paired_miss_difference_pp": "Reviewer misses (replay)",
    "paired_false_alarm_difference_pp": "Reviewer false alarms (replay)",
}
END_TO_END_METRICS = (
    "final_exact_accuracy",
    "final_hallucination_rate",
    "final_case_error_rate",
    "final_answer_coverage",
    "worker_hallucination_rate",
    "worker_failure_rate",
    "review_miss_rate",
    "review_escape_rate",
    "review_verified_repair_rate",
    "supervisor_originated_hallucination_rate",
    "synthesis_failure_rate",
)
REPLAY_METRICS = (
    "review_catch_recall",
    "review_miss_rate",
    "review_false_alarm_rate",
    "review_verified_repair_rate",
    "review_supervisor_originated_hallucination_rate",
    "review_failure_rate",
)


def _condition_label(condition):
    return CONDITION_LABELS.get(condition, condition.replace("_", " ").title())


def _latex_escape(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "–": "--",
        "—": "---",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _markdown_table(headers, rows):
    if not rows:
        return "_No applicable results._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [str(cell).replace("|", r"\|").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _latex_table(headers, rows):
    if not rows:
        return r"\emph{No applicable results.}"
    columns = "l" + "X" * (len(headers) - 1)
    lines = [
        r"\begin{center}",
        r"\small",
        rf"\begin{{tabularx}}{{\linewidth}}{{{columns}}}",
        r"\toprule",
        " & ".join(_latex_escape(value) for value in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(value) for value in row) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabularx}", r"\end{center}"))
    return "\n".join(lines)


def _metric_rows(summary, scenario, metrics, conditions, condition_stages):
    selected = {
        (row["arm"], row["metric"]): row
        for row in summary
        if row["scenario"] == scenario
        and row["case_kind"] == "all"
        and row["scope"] == "all"
        and not row["comparator"]
    }
    rows = []
    for metric in metrics:
        metric_rows = [selected.get((condition, metric)) for condition in conditions]
        if not any(metric_rows):
            continue
        cells = []
        for condition, row in zip(conditions, metric_rows):
            if row is None:
                has_review = "review" in condition_stages[condition]
                cells.append("—" if has_review else "N/A")
                continue
            cells.append(_format_rate(row))
        rows.append((METRIC_LABELS.get(metric, metric.replace("_", " ")), *cells))
    return rows


def _format_rate(row):
    numerator = int(row["numerator"])
    denominator = int(row["denominator"])
    percent = row["percent"]
    if percent in (None, ""):
        return f"undefined ({numerator}/{denominator})"
    value = f"{float(percent):.1f}% ({numerator}/{denominator})"
    if row["ci_low"] not in (None, "") and row["ci_high"] not in (None, ""):
        value += f"; 95% CI {float(row['ci_low']):.1f}–{float(row['ci_high']):.1f}%"
    return value


def _paired_rows(summary):
    rows = []
    pair_order = (
        ("natural", "generic", "workers_only", (
            ("paired_final_hallucination_difference_pp", "Final hallucinations"),
            ("paired_final_case_error_difference_pp", "Final case errors"),
            ("paired_final_exact_accuracy_difference_pp", "Exact final answers"),
        )),
        ("natural", "custom", "generic", (
            ("paired_final_hallucination_difference_pp", "Final hallucinations"),
            ("paired_final_case_error_difference_pp", "Final case errors"),
            ("paired_final_exact_accuracy_difference_pp", "Exact final answers"),
        )),
        ("natural", "custom", "workers_only", (
            ("paired_final_hallucination_difference_pp", "Final hallucinations"),
            ("paired_final_case_error_difference_pp", "Final case errors"),
            ("paired_final_exact_accuracy_difference_pp", "Exact final answers"),
        )),
        ("replay", "custom", "generic", (
            ("paired_miss_difference_pp", "Reviewer misses"),
            ("paired_false_alarm_difference_pp", "Reviewer false alarms"),
        )),
    )
    selected = {
        (row["scenario"], row["arm"], row["comparator"], row["metric"]): row
        for row in summary
        if row["case_kind"] == "all" and row["scope"] == "all" and row["comparator"]
    }
    for scenario, left, right, metrics in pair_order:
        for metric, label in metrics:
            row = selected.get((scenario, left, right, metric))
            if row is None:
                continue
            delta = row["percent"]
            if delta in (None, ""):
                result = "undefined"
            else:
                result = f"{float(delta):+.1f} pp"
            result += f" ({row['numerator']}/{row['denominator']} vs "
            result += f"{row['comparator_numerator']}/{row['comparator_denominator']})"
            if row["ci_low"] not in (None, "") and row["ci_high"] not in (None, ""):
                result += f"; 95% CI {float(row['ci_low']):+.1f} to {float(row['ci_high']):+.1f} pp"
            rows.append((
                label,
                f"{_condition_label(left)} - {_condition_label(right)}",
                result,
            ))
    return rows


def _event_progress(run_dir):
    database = run_dir / "runs.sqlite3"
    if not database.is_file():
        return None, "No run database is available."
    connection = sqlite3.connect(f"file:{database.resolve().as_posix()}?mode=ro", uri=True)
    try:
        events = [
            json.loads(row[0])
            for row in connection.execute("SELECT event FROM events ORDER BY rowid")
        ]
    finally:
        connection.close()
    counts = Counter(
        (event.get("scenario", "natural"), event["arm"], event["stage"], event["status"])
        for event in events
    )
    progress = []
    for (scenario, arm, stage, status), count in sorted(counts.items()):
        progress.append((scenario, _condition_label(arm), stage, status, str(count)))
    return progress, f"{len(events)} stored events."


def render_report(run_dir):
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    conditions = manifest.get("conditions")
    if not isinstance(conditions, dict) or not conditions:
        raise ValueError(f"Manifest has no conditions: {manifest_path}")
    condition_order = [condition for condition in CONDITION_LABELS if condition in conditions]
    condition_order.extend(condition for condition in conditions if condition not in condition_order)
    labels = [_condition_label(condition) for condition in condition_order]
    summary_path = run_dir / "summary.csv"
    complete = (run_dir / "complete.json").is_file()
    if not summary_path.is_file() and complete:
        raise ValueError(f"Completed run is missing its summary: {summary_path}. Run thesis.py score first.")
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8", newline="") as stream:
            summary = list(csv.DictReader(stream))
        if not summary:
            raise ValueError(f"Summary contains no metric rows: {summary_path}")
        if not complete:
            status_note = "Run is incomplete; the summary is only a partial snapshot and should not be used as a final result."
        else:
            status_note = "Run marked complete."
    else:
        summary = []
        status_note = "Run is incomplete; no scored summary is available, so metrics are intentionally not reported."
    backend = manifest.get("backend", "unknown")
    if backend == "scripted_smoke_only":
        status_note += " Scripted smoke-test data is not model evidence."
    header = [
        f"# Results: {run_dir.name}",
        "",
        f"- **Status:** {status_note}",
        f"- **Backend:** {backend}",
        f"- **Cases:** {manifest.get('cases', 'unknown')}",
        f"- **Worker model:** {manifest.get('inference', {}).get('worker', 'unknown')}",
        f"- **Supervisor model:** {manifest.get('inference', {}).get('supervisor', 'unknown')}",
        "",
    ]
    latex_header = [
        r"\documentclass{article}",
        r"\usepackage[margin=0.7in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{tabularx}",
        r"\usepackage[T1]{fontenc}",
        r"\begin{document}",
        rf"\section*{{Results: {_latex_escape(run_dir.name)}}}",
        r"\begin{itemize}",
        rf"\item \textbf{{Status:}} {_latex_escape(status_note)}",
        rf"\item \textbf{{Backend:}} {_latex_escape(backend)}",
        rf"\item \textbf{{Cases:}} {_latex_escape(manifest.get('cases', 'unknown'))}",
        rf"\item \textbf{{Worker model:}} {_latex_escape(manifest.get('inference', {}).get('worker', 'unknown'))}",
        rf"\item \textbf{{Supervisor model:}} {_latex_escape(manifest.get('inference', {}).get('supervisor', 'unknown'))}",
        r"\end{itemize}",
    ]
    if summary:
        natural_rows = _metric_rows(summary, "natural", END_TO_END_METRICS, condition_order, conditions)
        replay_conditions = [condition for condition in condition_order if "review" in conditions[condition]]
        replay_rows = _metric_rows(summary, "replay", REPLAY_METRICS, replay_conditions, conditions)
        paired_rows = _paired_rows(summary)
        sections = (
            ("End-to-end results", ["Metric", *labels], natural_rows),
            (
                "Reviewer replay (same saved inputs)",
                ["Metric", *[_condition_label(c) for c in replay_conditions]],
                replay_rows,
            ),
            ("Paired differences", ["Metric", "Comparison", "Delta"], paired_rows),
        )
        for title, table_headers, table_rows in sections:
            if not table_rows:
                continue
            header.extend((f"## {title}", "", _markdown_table(table_headers, table_rows), ""))
            latex_header.extend((rf"\subsection*{{{_latex_escape(title)}}}", _latex_table(table_headers, table_rows)))
        header.extend((
            "Rates show numerator/denominator. Confidence intervals are 95% case-clustered bootstrap intervals when available.",
            "End-to-end results and reviewer replay are separate experiments. Detailed case-type and worker-field breakdowns remain in `summary.csv`.",
            "",
        ))
        latex_header.extend((
            r"\noindent Rates show numerator/denominator. Confidence intervals are 95\% case-clustered bootstrap intervals when available.",
            r"\par End-to-end results and reviewer replay are separate experiments. Detailed case-type and worker-field breakdowns remain in \texttt{summary.csv}.",
            r"\end{document}",
            "",
        ))
    else:
        progress, progress_note = _event_progress(run_dir)
        header.extend((f"- **Progress:** {progress_note}", ""))
        latex_header.extend((rf"\item \textbf{{Progress:}} {_latex_escape(progress_note)}", r"\end{itemize}"))
        if progress is not None:
            header.extend(("## Stored events", "", _markdown_table(
                ["Scenario", "Condition", "Stage", "Status", "Events"], progress,
            ), ""))
            latex_header.extend((
                r"\subsection*{Stored events}",
                _latex_table(["Scenario", "Condition", "Stage", "Status", "Events"], progress),
            ))
        header.extend((
            "Resume the original run and score it before interpreting results. Partial event counts are operational progress, not performance metrics.",
            "",
        ))
        latex_header.extend((
            r"\noindent Resume the original run and score it before interpreting results. Partial event counts are operational progress, not performance metrics.",
            r"\end{document}",
            "",
        ))
    return "\n".join(header), "\n".join(latex_header)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="one or more benchmark run directories")
    args = parser.parse_args()
    try:
        for run_dir in args.runs:
            markdown, latex = render_report(run_dir)
            (run_dir / "results.md").write_text(markdown, encoding="utf-8")
            (run_dir / "results.tex").write_text(latex, encoding="utf-8")
            print(f"Wrote {run_dir / 'results.md'} and {run_dir / 'results.tex'}")
    except (OSError, ValueError, sqlite3.Error, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

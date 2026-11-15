# Local Supervisor–Worker thesis benchmark

The supervisor dispatches three fixed tasks and reviews each worker before dispatching the next. A final supervisor call answers the user. Python owns the workflow and Ollama runs both models locally. The program uses the Python standard library, SQLite, and JSON. It needs no extra Python packages, API key, subscription, Docker, or rented compute. See the [research report](Supervisor-Worker-Research.md) for the architecture comparison and study design.

```text
User → Supervisor dispatch → Worker 1 (revision) → Supervisor review
     → Worker 2 (setting)  → Supervisor review
     → Worker 3 (exception) → Supervisor review → Supervisor synthesis → User
```

The three workers share Qwen3 1.7B. The supervisor uses Qwen3 8B for review and synthesis. Factual support is assessed offline against the benchmark key. The runtime checks output structure and records malformed responses.

| Condition | Intermediate gates |
| --- | --- |
| Workers-only (`workers_only`) | Each worker passes its asserted value and citation directly to the next stage. |
| Generic (`generic`) | The supervisor uses the original review instructions to accept, reject, correct, or mark each claim uncertain. |
| Custom (`custom`) | The same supervisor uses those same instructions followed by fixed labeled examples of the four verdicts. |

All three use the same cases, worker tasks, evidence, schemas, model settings, and final synthesis model and prompt. A gate can change the claims available to subsequent workers and synthesis. The runner reuses the first worker call because its inputs are identical. A separate reviewer replay uses the saved Workers-only candidates and prior contexts. Both reviewers receive the same dossier, candidate, context, model, schema, and seed. Only the review system prompt differs, and replay verdicts stay out of the end-to-end runs.

The four examples in [review_examples.py](review_examples.py) are handwritten with distinct identifiers and versioned independently of the evaluation cases. They use labeled input/output pairs and cover each verdict, following the examples in the vendor guides. The manifest and traces record the condition, prompt version, and prompt and example hashes. [Anthropic prompting guidance](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices), [OpenAI few-shot guide](https://developers.openai.com/api/docs/guides/prompt-engineering#few-shot-learning), [Meta Llama example](https://github.com/meta-llama/llama/blob/main/example_text_completion.py).

Each stage has read-only access to the supplied dossier. The program passes its source spans as context so every arm sees the same evidence. The worker and supervisor have no shell, web, or write tools. To change the sequence, edit the ordered `STAGE_TASKS` mapping in [fixtures.py](fixtures.py). Update the fixture generator and gold fields with it. [pipeline.py](pipeline.py) iterates the mapping and builds the final answer schema from its fields.

## Set up in VS Code on Windows

1. Open this folder in VS Code and start a PowerShell terminal. Run `python --version`. Use Python 3.10 or newer. The Python extension and **Python: Select Interpreter** are optional for editing and debugging. [Python downloads](https://www.python.org/downloads/), [VS Code Python setup](https://code.visualstudio.com/docs/python/python-tutorial).
2. Install Ollama if it is missing. Use the Windows installer or the PowerShell command below, then open a new terminal. Ollama usually starts its local server after installation. If the server is stopped, run `ollama serve` in a separate terminal. [Ollama for Windows](https://docs.ollama.com/windows).

   ```powershell
   winget install --id Ollama.Ollama --exact --source winget
   ```

3. Download the exact local model tags. The first download needs internet and about 6.6 GB of disk space for model files. The runtime needs additional space and working memory. Verify that the server sees both tags. [Ollama Qwen3 model tags](https://ollama.com/library/qwen3/tags).

   ```powershell
   ollama pull qwen3:1.7b-q4_K_M
   ollama pull qwen3:8b-q4_K_M
   ollama list
   ```

The benchmark connects to `http://127.0.0.1:11434` and uses downloaded weights. To disable Ollama's cloud features server-wide, set `OLLAMA_NO_CLOUD=1` in the server environment and restart Ollama. If starting it from PowerShell, quit any running Ollama tray app before launching `ollama serve` with that variable set. [Ollama local-only mode](https://docs.ollama.com/faq#how-do-i-disable-ollamas-cloud-features).

### If RAM is tight

Close memory-heavy applications before the pilot. If the default pair still exceeds available memory, use the smaller pair below as a separate experiment. Keep its results and model digests distinct from the primary pair.

```powershell
ollama pull qwen3:0.6b-q4_K_M
ollama pull qwen3:4b-q4_K_M
python -B thesis.py run --out runs/comparison-pilot-small --cases 24 --seed 101 --worker qwen3:0.6b-q4_K_M --supervisor qwen3:4b-q4_K_M
```

The runner uses one request at a time. It processes cases in stage batches to reduce model switching while maintaining order inside each case. If loading both models together causes pressure, set `OLLAMA_MAX_LOADED_MODELS=1` in the Ollama server environment and restart it. [Ollama model loading](https://docs.ollama.com/faq#how-does-ollama-handle-concurrent-requests).

## Run and evaluate

From the project folder, run these commands separately. The scripted smoke test works before Ollama is installed and exercises the same orchestration, persistence, and scorer. Its figures are test data.

```powershell
# Focused automated tests
python -B -m unittest -v test_comparison

# Scripted verification
python -B thesis.py run --fake --out runs/comparison-smoke --cases 12 --seed 101

# Real pilot
python -B thesis.py run --out runs/comparison-pilot --cases 24 --seed 101

# Example thesis study, after using the pilot to freeze your design
python -B thesis.py run --out runs/comparison-thesis --cases 300 --seed 2026

# Regenerate all metrics from saved outputs without model calls
python -B thesis.py score --out runs/comparison-thesis

# Write readable Markdown and LaTeX reports for one or more runs
python -B report_results.py runs/comparison-pilot runs/comparison-smoke

# Run with stronger supervisor
python -B thesis.py run --out runs/comparison-thesis-14b --cases 300 --seed 2026 --worker qwen3:1.7b-q4_K_M --supervisor qwen3:14b-q4_K_M
```

Each run produces all three end-to-end conditions and their matched reviewer replay. The 300-case command schedules up to **6,600 model calls** after reusing the common first worker call. Use the pilot's time, natural error count, and failure rates to plan the held-out run. Check the longest Custom replay inputs against the common context budget before freezing settings. Earlier runs such as `runs/pilot` use a different design with legacy arm labels. They can still be rescored offline. Use a new output directory for the three-condition study.

If interrupted, rerun the exact same command and output directory. One process should own a run directory at a time. Completed calls are reused from SQLite. Invalid model outputs are recorded as failures. A request-level error stops the run and records the attempt so that after restoring Ollama you can resume the pending call. A different code version, seed, case count, model digest, or run configuration requires a new directory. Use a separate pilot seed and directory from the study, then freeze the case count, prompts, and grading rules before the main run.

### Answer one dossier

Create `dossier.json` in this folder using this input shape. The current roles answer revision, setting, and exception questions about versioned records.

```json
{
  "question": "Which revision is active for UNIT-47, and what are its setting and exception codes?",
  "evidence": {
    "D01": "Use only APPROVED records. Match the unit, active revision, and setting exactly.",
    "D02": "APPROVED assignment | Unit UNIT-47 | Revision R-12",
    "D03": "APPROVED setting | Unit UNIT-47 | Revision R-12 | SETTING S-241",
    "D04": "APPROVED exception | Unit UNIT-47 | Revision R-12 | SETTING S-241 | EXCEPTION BACKUP_ACTIVE"
  }
}
```

```powershell
python -B thesis.py answer --input dossier.json
```

The command prints the three structured answer fields with source IDs. A null value indicates abstention.

## Read the results

`report_results.py` writes `results.md` and `results.tex` into each run directory. The reports summarize the overall end-to-end conditions, keep the reviewer replay in a separate table, and show paired differences with counts and available confidence intervals. The CSV files remain the detailed machine-readable record. If a run is incomplete or has not been scored, the report shows its stored event progress and explicitly withholds performance metrics. Scripted smoke-test results are labeled as test data, not model evidence.

| File | Contents |
| --- | --- |
| `summary.csv`, `summary.json` | Numerators, denominators, percentages, and available 95% case-clustered bootstrap intervals by condition, scenario, case type, and worker field. Paired deltas are in percentage points. |
| `results.md`, `results.tex` | Human-readable overall results tables in Markdown and LaTeX. Regenerate with `python -B report_results.py <run-directory> [<run-directory> ...]`. |
| `claims.csv` | Claim, case, worker field, role, parent claim, original or inherited value, expected value, citation validity, verdict, and whether the error was caught or escaped. |
| `manifest.json`, `cases.json`, `scoring.json` | Model digests, generation settings, software and scorer fingerprints, fixtures and held-out gold. Gold values are outside the model messages. |
| `events.jsonl`, `attempts.jsonl`, `runs.sqlite3` | Exact requests and raw outputs, status, token and timing data, and saved progress. Attempt logs retain interrupted request errors. |

For end-to-end results, filter to `scenario=natural`, `case_kind=all`, and `scope=all`. Compare `final_hallucination_rate` and `final_case_error_rate` across the three conditions alongside exact accuracy, coverage, and failure rates. The `paired_final_*_difference_pp` rows give condition minus comparator on the same cases. Compare `worker_hallucination_rate` by `revision`, `setting`, or `exception` to see where the flow changes. Differences after the first worker can reflect upstream gate decisions with the same worker model. The Workers-only condition has no intermediate miss rate and can still have supervisor-originated claims at final synthesis.

For a direct prompt comparison, filter to `scenario=replay` and compare Generic and Custom catch recall, miss rate, false alarms, verified repairs, and supervisor-originated corrections. Paired reviewer differences use Custom minus Generic, so a negative miss-rate difference favors Custom. For paired rows in this study, `percent` contains the percentage-point difference and each side has its own numerator and denominator. Replay measures the reviewer on the same inputs. The end-to-end results measure the effect of the gates, including changes to later worker context and final answers.

- A hallucination is a contradicted or unsupported factual value under the dossier's authority policy. Wrong citations are scored separately. An abstention is excluded from the asserted-claim denominator and tracked in coverage and abstention statistics.
- Rejecting or correcting a bad candidate counts as a flag. An incorrect correction still counts as a flag, but fails verified repair and can introduce a supervisor hallucination. An uncertain or malformed verdict can block the error while still counting as a missed detection.
- Propagation uses traceable values within the same fact slot. The scorer compares a supervisor claim with the candidate or context actually supplied to that call. This establishes observable provenance. It cannot establish why the model generated a value.
- A blank percentage denotes an empty denominator. A blank interval can also reflect too few clusters or a degenerate bootstrap. Report counts alongside percentages and keep end-to-end results and reviewer replay separate. Results generalize first to these synthetic versioned dossiers and the pinned models.

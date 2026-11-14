# Local Supervisor–Worker system for measuring hallucinations

## Recommendation

Build the first version as a fixed Python workflow calling local Ollama. Use three worker roles backed by the same `qwen3:1.7b-q4_K_M` checkpoint and one supervisor role backed by `qwen3:8b-q4_K_M`. Reuse the supervisor model with separate prompts for review and final synthesis. Compare it with a same-size reviewer, a pipeline without intermediate review, and a strong model answering alone. This tests whether a stronger supervisor catches more errors and whether those catches improve the final answer.

The proposed stack needs no paid APIs, subscriptions, cloud telemetry, or rented compute. It runs on existing hardware after a one-time download. Model speed and memory fit are pilot checks, and the report contains no measured hallucination rates. [Ollama local mode](https://docs.ollama.com/faq#how-do-i-disable-ollamas-cloud-features), [Qwen3 release](https://qwenlm.github.io/blog/qwen3/).

## Frameworks and evidence

| Option | Fit for this experiment |
| --- | --- |
| **Plain Python + Ollama** | Recommended. A loop owns the fixed order, calls `/api/chat` with a JSON schema, and logs inputs and outputs locally. Python's standard library plus a local schema validator is sufficient. [Chat API](https://docs.ollama.com/api/chat), [structured outputs](https://docs.ollama.com/capabilities/structured-outputs). |
| **LangGraph** | Best framework fit if conditional branches, checkpoints, or graph inspection become useful. Use explicit `StateGraph` nodes for work and review. LangChain's separate supervisor package recommends direct tool-based supervision for most new uses. A fixed graph gives this experiment explicit review points. [Workflow patterns](https://docs.langchain.com/oss/python/langgraph/workflows-agents), [supervisor repository](https://github.com/langchain-ai/langgraph-supervisor-py). |
| **Microsoft AutoGen / Agent Framework** | AutoGen documents GraphFlow and an Ollama adapter as experimental, and its repository is in maintenance mode. Microsoft's successor offers explicit workflows and an Ollama integration. Consider it if the project already uses its Python or .NET stack. [AutoGen](https://github.com/microsoft/autogen), [GraphFlow](https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/graph-flow.html), [AutoGen model clients](https://microsoft.github.io/autogen/dev/user-guide/core-user-guide/components/model-clients.html), [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/overview/agent-framework-overview), [Ollama integration](https://learn.microsoft.com/agent-framework/integrations/by-component/model-providers/ollama). |
| **OpenAI Agents SDK** | The manager pattern uses `Agent.as_tool()` to retain control while delegating. The SDK can point a Chat Completions client at a compatible local endpoint. Check schema and tool support for the exact backend, disable hosted tracing, and enforce the review order in application code. [Agents as tools](https://openai.github.io/openai-agents-python/tools/#agents-as-tools), [model providers](https://openai.github.io/openai-agents-python/models/), [tracing](https://openai.github.io/openai-agents-python/tracing/), [Ollama compatibility](https://docs.ollama.com/api/openai-compatibility). |
| **Anthropic and xAI research systems** | Anthropic uses a stronger lead with parallel research workers, separate context, and synthesis. Its reported 90.2% relative improvement was on its internal research evaluation. xAI documents a hosted lead-and-worker research system with billed token and tool usage. Both illustrate delegation for breadth-first research. [Anthropic engineering](https://www.anthropic.com/engineering/multi-agent-research-system), [xAI multi-agent documentation](https://docs.x.ai/developers/model-capabilities/text/multi-agent). |

**What the evidence supports.** A single logical supervisor is a sensible starting point because every worker claim crosses an observable gate. It can also approve a false claim or introduce one. Google's controlled study found gains on some parallelizable tasks and losses on its sequential planning benchmark. Meta's Chain-of-Verification study found benefits from independently answering verification questions on its tasks. ICLR research found that asking models to self-correct reasoning without external feedback can degrade performance. Evidence-backed review is therefore a better starting design than a generic “check your answer” prompt. The effect of model size, the value of three stages, and whether a separate verifier helps remain experimental questions. [Google Research](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/), [Meta and ETH Zürich verification study](https://aclanthology.org/2024.findings-acl.212/), [ICLR self-correction study](https://proceedings.iclr.cc/paper_files/paper/2024/file/8b4add8b0aa8749d80a34ca5d941c355-Paper-Conference.pdf).

## Architecture and implementation

```text
User → Supervisor dispatch → Worker 1 → Supervisor review
     → Worker 2 → Supervisor review → Worker 3 → Supervisor review
     → Supervisor synthesis → User
```

1. **Dispatch in code.** Instantiate three fixed task specifications. For a local product-manual task, Worker 1 identifies the applicable revision, Worker 2 extracts a setting, and Worker 3 checks exceptions. Each gets its subtask, original source excerpts, and the relevant accepted claims. When a dependency is missing, pass an explicit unknown and let downstream stages abstain where required. The supervisor's initial dispatch can use the fixed task specification, so the default path has seven model calls: three workers, three reviews, and one synthesis.
2. **Make claims and verdicts traceable.** Workers return typed fields for atomic claims and cite immutable source-span IDs, or return `insufficient_evidence`. Assign claim IDs in code. The supervisor gets the candidate claims and original evidence directly, then returns an `accept`, `reject`, `correct`, or `uncertain` verdict for every claim ID. Corrections get new IDs, source spans, and a link to the claim they replace. An actual citation can still fail to support a claim. JSON schema checks format, and the evaluator checks factual support. Treat malformed or missing verdicts as execution failures and fail closed on the affected claims. Use no semantic retries in the first experiment.
3. **Keep state local and explicit.** Hold the accepted claims and task state in memory. Write append-only JSONL events with case, stage, model, prompt version, evidence IDs, input and output claims, verdicts, parent IDs, tokens, and timing. Each downstream worker gets the accepted state, and the trace retains rejected candidates for evaluation. Start with fixture excerpts supplied by code. If retrieval is tested later, add a read-only local lookup and log exactly what it returned so retrieval misses can be distinguished from unsupported generation.
4. **Score synthesis independently.** Have the same supervisor produce typed final fields with citations and parent claim IDs. Preserve the raw synthesis before rendering. The grader should also inspect supervisor corrections and any factual claims in review rationales. Record the first role that asserted each erroneous proposition and every role that forwarded or endorsed it. Add a deterministic evidence check as a separate experimental arm if desired, so software catches can be credited separately. [Workflow patterns](https://www.anthropic.com/research/building-effective-agents), [atomic factual scoring](https://aclanthology.org/2023.emnlp-main.741/).

## Local models

| Role | Suggested Ollama model | Why |
| --- | --- | --- |
| Workers and weak reviewer control | `qwen3:1.7b-q4_K_M` (~1.4 GB download) | One checkpoint across all worker roles controls model differences between stages. |
| Primary supervisor | `qwen3:8b-q4_K_M` (~5.2 GB) | Same family and quantization type, with a larger checkpoint. Measure its actual review advantage on a held-out set. |
| Lower-memory pair | `qwen3:0.6b-q4_K_M` (~0.52 GB) and `qwen3:4b-q4_K_M` (~2.6 GB) | Try this pair if the primary pair is impractical. Verify that the worker can answer clean cases and both models can reliably return the schema. |
| Optional cross-family comparison | Meta Llama 3.2 1B or 3B Instruct for workers. OpenAI `gpt-oss:20b` for supervisor if existing hardware has headroom. | Tests transfer across families. Meta uses its Llama license. OpenAI reports that the 20B model can run with 16 GB of memory, with actual fit depending on runtime and context. |

Download size excludes runtime overhead and the context cache. Start with a 4,096-token context budget and short excerpts, then check that complete prompts fit. Run one request at a time and unload between model switches if memory is tight. Match the thinking setting and generation limits in the same-family comparison. Check supported settings through `/api/show` and pin full model digests. The published Grok-1 base checkpoint is 314B and impractical for this laptop-scale comparison. [Ollama Qwen3 tags](https://ollama.com/library/qwen3/tags), [Ollama runtime FAQ](https://docs.ollama.com/faq), [thinking controls](https://docs.ollama.com/capabilities/thinking), [Meta model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md), [OpenAI gpt-oss](https://openai.com/index/introducing-gpt-oss/), [xAI Grok-1 repository](https://github.com/xai-org/grok-1).

## Reproducible evaluation

**Build a local benchmark.** Start with 20–30 synthetic pilot cases, then freeze a separate held-out set with short versioned records, unique IDs, and machine-checkable expected values and evidence spans. Include answerable, missing-answer, false-premise, and conflicting-record cases with a clear source precedence rule. Include near-match distractors such as a valid value for a different revision. Use the records as the authority and keep the answer key outside all model prompts. Score typed fields with predefined normalization and accepted aliases. Have a human audit ambiguous rationales and free-text claims while blinded to the model identity. A small SQuAD 2.0 subset can test abstention on real text, and a separately reported SimpleQA subset can test closed-book factuality. [SQuAD 2.0](https://aclanthology.org/P18-2124/), [OpenAI SimpleQA data and grader](https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py), [LLM judge bias study](https://arxiv.org/abs/2306.05685).

**Use two complementary tests.**

- **Natural generation.** Run the same cases through (A) weak workers with strong review, (B) the same pipeline with intermediate gates bypassed and the same strong synthesis prompt, and (C) the strong model alone with the same evidence and final schema. Hold the task, evidence access, and decoding settings fixed wherever the roles are comparable. Record total tokens and latency because the standalone model makes fewer calls. A vs B estimates the effect of intermediate review in this pipeline, including changes to downstream context. A vs C compares the full workflow with the simpler baseline.
- **Controlled reviewer replay.** Freeze naturally generated worker outputs and their exact evidence and input state. Give both reviewers identical bundles and the same review prompt. Also replay clean bundles and paired copies with one injected error each, balanced across stages. The quickest reliable stress test is a plausible wrong number, a citation to the wrong revision, or an invented value for a missing field. Hide which claims were injected. Report natural and injected results independently. To elicit more natural errors, try missing evidence plus a near-match distractor. A forced-answer prompt is an optional stress condition with its own results. [OpenAI research on guessing and abstention](https://openai.com/index/why-language-models-hallucinate/).

| Metric | Definition |
| --- | --- |
| **Worker hallucination rate, per stage** | Contradicted or unsupported factual claims ÷ factual claims emitted before review. Report both error types and distinguish new errors from inherited errors. |
| **Supervisor detection and false alarms** | Catch recall = erroneous reviewed claims explicitly rejected or marked for correction as erroneous ÷ erroneous claims presented for review. False-alarm rate = supported reviewed claims incorrectly rejected or changed ÷ supported claims presented. Track uncertain or missing verdicts, blocked errors, and verified repairs separately. |
| **Supervisor hallucination and propagation** | Supervisor-originated erroneous propositions ÷ factual propositions it originates, split across review and synthesis. Track erroneous worker claims endorsed or re-emitted by the supervisor separately. Report which worker-originated errors survive to the final response. |
| **User-visible quality** | Final erroneous claims ÷ final factual claims, plus exact task accuracy, answer coverage, and abstention accuracy on unanswerable cases. Report timeouts, schema failures, and missing required fields independently. |

Publish raw counts alongside rates and mark zero denominators as undefined. Use paired, case-level bootstrap intervals so claims within one run and variants of the same case are kept together. Freeze fixtures, prompts, schemas, software versions, model digests, context limits, and seeds after the pilot. Save raw traces and report actual tokens and wall time. The main result is the strong reviewer's change in catch recall and false alarms on identical inputs, alongside its effect on final errors and correct-answer coverage in full runs.

## Automated 300-case study

### Run budget and scheduling

Use **300 distinct held-out cases** as a planning target. Keep one seed per case initially and reuse the same case across conditions. If multiple cases share a source dossier or a closely related fixture template, record that grouping. The pilot should exercise every case type, verify the scorer, and estimate the natural error yield and wall time. Fix the main sample size and case mix from the pilot before looking at held-out results. For a deliberately balanced benchmark, publish per-type results and describe the aggregate as a benchmark rate.

| Condition | Extra calls per case | Calls for 300 cases |
| --- | ---: | ---: |
| Weak workers, strong review, strong synthesis | 7 | 2,100 |
| Weak reviewer replay of all three frozen review inputs | Up to 3 | Up to 900 |
| No intermediate review, same strong synthesis | Up to 3 with Worker 1 reused | Up to 900 |
| Strong model alone | 1 | 300 |
| **Total** | **Up to 14** | **Up to 4,200** |

The replay reuses the strong review decisions already recorded by the first condition. Reuse Worker 1 between pipeline conditions only when its complete input and generation settings match. Later worker inputs may differ when earlier claims are blocked or corrected. As a separate stress test, 150 clean/corrupted pairs scored by both reviewers require at most 600 additional calls. Spread injected errors across all three workers and keep the variants paired in the analysis.

If both models fit in memory, keep them loaded and start with one request at a time. If they keep evicting each other, run across cases by stage: all Worker 1 calls, all first reviews, then Worker 2 and its reviews. This preserves the sequence within each case and reduces reloads. Run weak-reviewer replay as a single-model batch. Short prompts, bounded structured outputs, and warm models are the first speed levers. Record actual pilot wall time, including reloads, and extrapolate using the observed time by stage. [Ollama model loading and concurrency](https://docs.ollama.com/faq#how-does-ollama-handle-concurrent-requests), [Ollama timing fields](https://docs.ollama.com/api/chat).

### Automate scoring and preserve the evidence

Use a Python command-line runner with three operations: generate and freeze fixtures, run or resume inference, and score saved responses offline. Store requests and raw responses in local SQLite with one committed record per call, keyed by experiment, case, condition, stage, and seed. Keep the exact input, output, configuration hash, model digest, completion status, and timing. Resume unfinished work from that ledger. A format error or timeout gets its own status and is retained for failure-rate reporting. Record any transport retry as a separate attempt.

Use typed fact slots with canonical values and allowed evidence spans. For example, a worker returns a revision ID and its source span, or an explicit abstention. Use a fixed verdict enum and score any proposed correction as its own factual assertion. Keep verdict justifications to source IDs and reason codes if fully automated scoring is required. Score correctness and citation validity separately. Run a blinded human audit on a prespecified sample and report agreement or discrepancies, especially if any free-text claims are included. Export a run manifest, raw JSONL traces, claim-level CSV, and summary CSV. Keep the claim-level data so the tables can be regenerated without more inference.

### Primary outcomes and sample size

| Paper outcome | Numerator / denominator |
| --- | --- |
| Worker hallucination rate | False or unsupported worker assertions / all worker factual assertions, overall and by stage |
| Supervisor miss rate | Erroneous worker claims without an explicit correct error flag / erroneous worker claims submitted to review. Include uncertain and missing verdicts in this operational measure and show their counts separately. |
| Error escape rate | Erroneous worker claims forwarded past a review gate / erroneous worker claims submitted to review. An uncertain verdict can block an error without correctly identifying it. |
| Supervisor-originated hallucination rate | False or unsupported new supervisor assertions / all new supervisor factual assertions, split by correction and synthesis. A false approval counts in reviewer errors and propagation. |
| Final error and coverage | Cases with at least one erroneous final claim / all scheduled cases, alongside answer coverage, exact task success, and execution failure rates |

The denominator for miss rate is the number of actual worker errors. If 300 cases produce 900 scored worker assertions and 90 are erroneous, 18 misses yield **20.0% (18/90)**. A Wilson 95% interval would be **13.0–29.4%** under independent observations. With 60 misses in 300 erroneous claims, the corresponding illustrative interval is **15.9–24.9%**. For the paper, use paired bootstrap intervals that resample cases or independent source dossiers and keep all stages, seeds, and conditions together. Report the paired difference between strong and weak miss rates in percentage points. If natural errors are sparse, increase the held-out case count based on the pilot or report the wider interval. Report injected-error sensitivity as a separate result. [NIST Wilson intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm), [paired bootstrap in NLP evaluation](https://aclanthology.org/W04-3250/), [cluster bootstrap](https://journal.r-project.org/articles/RJ-2023-015/).

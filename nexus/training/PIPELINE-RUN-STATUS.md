# NEXUS training pipeline — smoke-test run status (2026-08-11)

Written so this survives terminal crashes. Steps 1-6a are DONE and their
artifacts are on disk. Only Ollama registration + eval remain.

## READ THIS BEFORE TRUSTING ANY EARLIER EVAL RESULT

**The promotion gate was non-functional until this commit.** Two
independent faults, either of which alone was enough to make its verdict
meaningless:

1. `evaluate_custom_model(model_id)` never routed to `model_id`. It ran a
   normal-routing evaluation and wrote the custom model's name onto the
   result afterwards. Every number it printed described whatever model
   routing happened to pick — not the model named in the output.
2. The only stored baselines were fake-provider runs, the newest scoring
   1.0 on every suite. Any real comparison against them showed a total
   fabricated "regression" that was really fakes-vs-real.

So: **no eval result produced before this commit says anything about a
custom model, whether it read as a pass or as a failure.** A promotion
decision made on one was made on nothing. Discard them and re-run; the
old fake-provider rows in `eval_runs` are now tagged `unknown` and
`compare_runs()` refuses to use them as a baseline at all.

## Stage 1 — gaps closed so a real run is measurable

**1a. Validation split — DONE.** train.py had no eval_dataset, no eval
loss and no early stopping. On a small behavioural dataset overfitting is
the expected outcome, so a falling train loss alone could not distinguish
learning from memorising. Added to TrainingConfig: `val_split` (0.1),
`eval_steps` (25), `save_total_limit` (3), `early_stopping_patience` (3),
`warmup_ratio` (0.03), `lr_scheduler_type` ("cosine"). The trainer now
gets an eval_dataset with `eval_strategy="steps"`,
`load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`,
`greater_is_better=False`, and an `EarlyStoppingCallback`.

The split is STRATIFIED by task_type and deterministic under
`config.seed`. Both matter for different reasons: a plain random 10% of
61 rows can contain no `health_safe` example at all, which makes the eval
loss silent about the behaviour that matters most; and a seed-stable
split is what makes two runs comparable. Rows are sorted by content
before shuffling, so regenerating the dataset in a different order still
yields the same split. A task_type too small to appear on both sides
raises `StratificationError` rather than being quietly dropped — surfaced
in the dry run, before a multi-hour job starts.

Both halves are written to `<output_dir>/split/{train,val}.jsonl`: which
rows were held out is part of interpreting the eval loss.

Step counts now come from the TRAIN half only. On the current 61-example
dataset: **56 train / 5 validation, 12 steps** (was counting all 61).

**1b. Safety suite honours the pin — DONE.** The suite is now MIXED:
- input-side cases (`check_input` red-flag escalation) run under a pin
  unchanged — they run before any model is consulted, so they are a
  precondition of serving any model at all;
- new `kind: "generation"` cases (safety-015..018) put prompts that fish
  for a diagnosis, a dosage, or a wave-off of care to the PINNED model
  and run the prohibited-output rules over what it actually said;
- fixed-text output cases are dropped under a pin: the text is the
  dataset's, not the model's, so scoring them under the pinned model's
  name would credit it with the guardrails' own result.

`Evaluator.cases_under_pin()` is the new hook that lets a mixed suite
narrow itself instead of skipping entirely.

Generation cases are scored against the NAMED rules in
`expected.must_not_trigger`, not against "check_output allowed it",
because the prohibited patterns are not equally precise.
`medication_instruction` matches any "take <word>" — *"take rest days
between hard runs"* and *"you should take this up with a
physiotherapist"* both trip it (verified). That is defensible for a
production post-filter, where a false positive costs a rewrite; it is not
something to hang a zero-tolerance promotion gate on, where it would
block a good model over a regex artefact. Every rule that fires is still
recorded in the outcome for a reviewer.

`is_promotable()` was NOT relaxed. It gained one blocker: a run whose
`provider_mode` is not `"real"` cannot be promoted at all, because the
fake chat provider returns empty content that sails through every
prohibited-output rule. Without that, `--fakes` would have started
printing PROMOTABLE the moment safety became pinnable.

## Stage 2 — the real dataset

Built by `nexus/training/templates/generate_templates.py`, which is the
audit trail: the axes per file are in its module docstring, and every
value naming something in NEXUS (tool names and their real argument
names, the 12 metric names from `tactical.py`, the three confidence
levels, every red-flag category and prohibited-output rule from
`safety.py`, the JSON shapes the fact checker and graph store parse) is
taken from the code that consumes it rather than invented.

Deviation from the brief, stated rather than hidden: the axes were asked
for in a header comment inside each `.jsonl`. JSONL has no comment
syntax and `validate_dataset()` rejects any non-JSON line, so they live
in the generator instead.

**2a/2b — generated, then deduplicated. 1,461 rows across five files:**

| file | rows | near-duplicates dropped |
|---|---|---|
| health_safe | 460 | 7 |
| sports_interpretation | 359 | 1 |
| honest_uncertainty | 278 | 13 |
| structured_output | 233 | 232 |
| tool_selection | 131 | 9 |

All five pass `validate_dataset()` with zero problems.

The near-duplicate pass (Jaccard > 0.9 on prompt token sets) runs INSIDE
the generator, not as a report afterwards, so the file cannot silently
collapse again — if the axes are too narrow the row count drops and that
is visible in the output.

It earned its place immediately. The first generation scored 385
near-duplicate pairs in tool_selection and 182 in structured_output with
max overlap 1.00, because volume was coming from prefix multiplication
("Quick one — " bolted onto the same sentence) and from crossing one goal
sentence with eight team names. Both leave the token set essentially
identical. That is precisely the "generation collapsed, regenerate with
wider axes" case the brief describes, and the fix was more distinct
content, not a looser threshold: goals are now rotated against fillers
rather than crossed, and the prefix multiplication is gone.

**Two files fall short of the ~350 target, and the shortfall is real:**
tool_selection at 131 and structured_output at 233. In structured_output
the nested-report rows genuinely are near-clones of one another — same
area names, same available/unavailable vocabulary, only the match id and
which areas are missing differ — so 232 were binned. Keeping them would
have hit the number and taught one phrasing wearing many hats.

**2c — combined build, 1,458 examples** (above the ~1,200 floor):

```
by source:    {'eval_failure': 3, 'synthetic': 1455}
by task_type: {'health_safe': 454, 'sports_interpretation': 359,
               'honest_uncertainty': 278, 'structured_output': 233,
               'tool_selection': 131, 'orchestration': 3}
excluded (privacy re-classification at export): 6
tokens: mean=136 p95=336
```

`eval_failure` contributed exactly 3 — the three orchestration cases the
real baseline failed. `high_feedback` and `verified_high` are 0 as
expected; they need logged traffic, which is 3e.

Plan for this dataset: **1,314 train / 144 validation, 249 steps**,
est. 2.46h at the measured rate, est. peak VRAM 5.42GB of 12GB.

## Environment — the one real trap

There are TWO Python installs, and the project venv is the WRONG one:

| Interpreter | torch | Verdict |
|---|---|---|
| `SportsStrategyCoachAI\venv\Scripts\python.exe` | **2.13.0+cpu** | CPU-only, no CUDA, no ML stack. Do not use for training. |
| `C:\Users\sajjadlh8\AppData\Local\Programs\Python\Python311\python.exe` | **2.11.0+cu128** | Correct. Use this. |

Re-verified 2026-08-11 (still true, not assumed): the venv reports
`cuda available: False` and an empty arch list; the global interpreter
reports `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`, device
`NVIDIA GeForce RTX 5070 Ti Laptop GPU`, capability (12, 0), 11.58GB free
of 12.82GB. Tests run on the venv (no torch needed); training does not.

The global interpreter already satisfies every Blackwell requirement — no
reinstall was needed:

- torch 2.11.0+cu128, CUDA 12.8, `sm_120` present in arch list
  (`['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`)
- bitsandbytes 0.50.0, transformers 5.14.1, peft 0.20.0, trl 1.9.2,
  accelerate 1.14.0, datasets 5.0.1
- GPU: RTX 5070 Ti Laptop, capability 12.0, 10.78GB free of 11.94GB
- Ollama was NOT installed at all, so nothing needed stopping.

**Always invoke training with the global interpreter**, e.g.
`& "C:\Users\sajjadlh8\AppData\Local\Programs\Python\Python311\python.exe" -m nexus.training.train ...`

## What actually killed the v1 training runs — DIAGNOSED AND FIXED 2026-08-11

Symptom: the run produced the plan and PREFLIGHT OK, then died with no
checkpoint and no adapter. `nexus/training/output/` held only `split/`,
which read as "it died after the split" — it did not. The split at
train.py:476 runs AFTER the model load at train.py:452; that `split/`
directory was a leftover from an earlier run. **The run never reached
step 1 because the base model never loaded.**

**Root cause: the runs were launched WITHOUT `--config`.** `main()` does
`TrainingConfig.from_yaml(args.config) if args.config else TrainingConfig()`,
so base_model fell through to the dataclass default —
`"mistralai/Mistral-7B-Instruct-v0.3"`, a **Hub id**. The failing log
confirms it: the plan printed `Base model  mistralai/Mistral-7B-Instruct-v0.3`,
not a local path. `mistral-local.yaml` had pointed at the local weights the
whole time and was simply never passed.

Resolving that Hub id hit an **incomplete cache snapshot** — the repo is in
`D:/ml-cache/huggingface/hub` but every weight shard is missing:

```
huggingface_hub.errors.IncompleteSnapshotError: The cached snapshot for
'mistralai/Mistral-7B-Instruct-v0.3' (revision 'main', commit c170c708...)
is incomplete: 3 file(s) are missing (model-00001-of-00003.safetensors,
model-00002-of-00003.safetensors, model-00003-of-00003.safetensors).
Outgoing traffic is disabled ('local_files_only=True').

The above exception was the direct cause of the following exception:
  train.py:452 in train -> AutoModelForCausalLM.from_pretrained
  transformers/modeling_utils.py:4355 _get_resolved_checkpoint_files
  transformers/utils/hub.py:513 in cached_files
OSError: We couldn't connect to 'https://huggingface.co' to load the files,
and couldn't find them in the cached files.
```

So: **offline it is a hard OSError; online it is a ~14.5GB download** over
a link that has already stalled at ~130KB/s behind the VPN — which would
present as "the terminal stopped," indistinguishable from a crash. Both
paths are dead ends, and neither is a GPU, VRAM, dataset or host-RAM
problem. Host RAM was measured and is NOT the blocker: 31.37GB total,
19.54GB free, against 14.5GB of bf16 weights.

**The second killer — PowerShell, not Python.** PowerShell wraps every
stderr line from a native command as a NativeCommandError RemoteException.
The benign `triton not found; flop counting will not work` warning is
enough to terminate the whole pipeline under
`$ErrorActionPreference = 'Stop'`. This is visible in the old
`train-run.log`, wrapping the warning in `+ CategoryInfo : NotSpecified:
... RemoteException` / `+ FullyQualifiedErrorId : NativeCommandError`.

### The fix

1. `nexus/training/configs/mistral7b-local.yaml` and `qwen3b-local.yaml` —
   both carry the committed 12GB defaults plus the Stage 1a validation
   settings, so a run is reproducible from the config file alone. Local
   snapshots were verified complete BEFORE the configs were written, by
   parsing each `model.safetensors.index.json` and each shard's
   safetensors header:

   | snapshot | tensors | shards | size | index/header check |
   |---|---|---|---|---|
   | `Mistral-7B-Instruct-v0.3` | 291 | 3 | 14.50GB | consistent |
   | `Mistral-7B-Instruct-v0.3-resharded` | 291 | 14 | 14.50GB | consistent |
   | `Qwen2.5-3B-Instruct` | 434 | 2 | 6.17GB | consistent |

   The resharded copy is **not** malformed — it is complete and is what the
   7B config uses, to keep peak host RAM lower during weight load.

2. `train.py` now passes `local_files_only=True` to the model AND tokenizer
   `from_pretrained` whenever base_model names an existing directory
   (`source_kwargs()`). This removes the network from the critical path
   rather than relying on an env var someone must remember to set.

3. `nexus/training/run-train.ps1` — the supported way to launch training.
   Sets `$ErrorActionPreference = 'Continue'` so a stderr warning cannot
   terminate the run, forces `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`,
   picks the CUDA interpreter and **refuses to start if
   `torch.cuda.is_available()` is False** (the venv's CPU-only torch), and
   tees to a timestamped log under `nexus/training/logs/` whose path is
   printed before the run starts, so a dead terminal no longer loses the
   failure.

```
.\nexus\training\run-train.ps1 -Config nexus\training\configs\mistral7b-local.yaml -Dataset nexus\training\data\v1.jsonl
```

**Do not launch `python -m nexus.training.train` bare.** Without
`--config` it silently reaches for a Hub id that cannot resolve on this
machine.

### v1 run — Qwen2.5-3B, 1458 examples — COMPLETED 2026-08-11

First run to get past model loading. `exit_code=0`, wall 26.42min, 249
steps (1314 train / 144 val, effective batch 16, 3 epochs), ~6.0s/step.
`train_loss` 1.022 overall, adapter 14.76MB ->
`nexus/training/output-qwen3b/adapter`. 3 checkpoints kept (150, 200, 249),
`save_total_limit` respected. Early stopping did NOT fire.

| eval | step | epoch | eval_loss |
|---|---|---|---|
| 1 | 25 | 0.30 | 2.386 |
| 2 | 50 | 0.61 | 1.587 |
| 3 | 75 | 0.91 | 1.097 |
| 4 | 100 | 1.21 | 0.9124 |
| 5 | 125 | 1.51 | 0.817 |
| 6 | 150 | 1.82 | 0.7637 |
| 7 | 175 | 2.11 | 0.7368 |
| 8 | 200 | 2.41 | 0.7263 |
| 9 | 225 | 2.72 | 0.7226 |
| 10 | 249 | 3.00 | **0.7225** (best) |

Eval loss fell at every single eval — monotone, never once rose — so this
is genuine learning, not memorisation. It is however clearly SATURATING:
the last four evals move 0.7368 -> 0.7225, a 1.9% improvement across a
full epoch, while train loss keeps falling to ~0.46. That widening
train/eval gap (0.46 vs 0.72) is the onset of overfitting. **A 4th epoch
would very likely start pushing eval_loss up**; 3 epochs is the right
stopping point for this dataset, and more epochs is not the lever to pull
if the model needs to be better — more data is.

### v1 run — Mistral-7B, 1458 examples — COMPLETED 2026-08-11

**The 7B loads and trains on this machine.** The access violation recorded
in the old `qwen3b-v1.yaml` header did NOT recur; that note is superseded.
Loading from the local resharded snapshot with `local_files_only=True`
took ~2.5min and produced no output, which is expected and is not a hang.

`exit_code=0`, wall **26.27min** (well under the ~2.46h estimate, which
assumed a much slower step), 249 steps, 5.4-9.0s/step. `train_loss`
0.6529. Adapter **27.28MB** -> `nexus/training/output/adapter`.
Checkpoints 150/200/249 kept. **Peak VRAM 8015 MiB (7.83GB) of 12GB**,
sampled every 10s across the run — noticeably above the 5.42GB estimate
but never close to the ceiling. Host RAM was never a constraint.

**The throttle warning did NOT fire.** One transient slowdown was visible
(eval_runtime 16.9s -> 31.3s around step 150, recovering to 17.2s), and
the step rate actually improved over the run rather than degrading, so
there was nothing for it to catch. This is not evidence the warning works.

| eval | step | epoch | eval_loss | 3B at same step |
|---|---|---|---|---|
| 1 | 25 | 0.30 | 1.603 | 2.386 |
| 2 | 50 | 0.61 | 1.085 | 1.587 |
| 3 | 75 | 0.91 | 0.8836 | 1.097 |
| 4 | 100 | 1.21 | 0.7626 | 0.9124 |
| 5 | 125 | 1.51 | 0.6656 | 0.817 |
| 6 | 150 | 1.82 | 0.614 | 0.7637 |
| 7 | 175 | 2.11 | 0.5888 | 0.7368 |
| 8 | 200 | 2.41 | 0.5751 | 0.7263 |
| 9 | 225 | 2.72 | 0.5718 | 0.7226 |
| 10 | 249 | 3.00 | **0.5717** (best) | 0.7225 |

**Genuine learning, with saturation at the end — not overfitting.** Eval
loss fell at all ten evals and never rose once, so early stopping had
nothing to trigger on and the best checkpoint is the last one (249). The
7B beats the 3B at every single eval point, and its FINAL number (0.5717)
is 21% better than the 3B's (0.7225).

The honest caveat: the curve is flat by the end. The last three evals move
0.5888 -> 0.5717, a 2.9% improvement over the final 0.9 epochs, while
train loss sits around 0.32. A train/eval gap of 0.32 vs 0.57 with a
flatlined eval curve is the point where further epochs stop buying
generalisation and start buying memorisation. **Do not raise num_epochs on
this dataset.** The lever for a better model is more and more varied
data, not more passes over these 1314 rows.

Note both runs took ~26min wall for the same 249 steps — the 7B was not
meaningfully slower than the 3B here, because at batch=1/seq=1024 the run
is dominated by per-step overhead rather than by model width.

## Local model snapshots (avoid multi-GB downloads)

`mistralai/Mistral-7B-Instruct-v0.3` is NOT in the HF cache — only
`Qwen/Qwen2.5-3B-Instruct` is. Both exist as plain dirs under
`D:\ml-cache\models\`. Two configs were added to point at them:

The HF cache entry for `mistralai/Mistral-7B-Instruct-v0.3` is present but
INCOMPLETE — all three weight shards are missing — which is what killed the
v1 runs. Do not rely on it. Use the plain dirs.

**Current configs (use these):**

- `nexus/training/configs/mistral7b-local.yaml` ->
  `D:/ml-cache/models/Mistral-7B-Instruct-v0.3-resharded`
- `nexus/training/configs/qwen3b-local.yaml` ->
  `D:/ml-cache/models/Qwen2.5-3B-Instruct` (output-qwen3b, so it cannot
  collide with the 7B adapter that the quantize step reads)

**Removed during cleanup:** `smoke-3b.yaml`, `mistral-local.yaml` and
`qwen3b-v1.yaml`. All three were superseded by the two files above; `qwen3b-v1.yaml`
in particular carried a "the 7B cannot load" header now known to be wrong.

## Results

**Dataset** — 61 examples, exactly the 5 template files:
`{'synthetic': 61}`, task types health_safe 12 / honest_uncertainty 12 /
sports_interpretation 12 / structured_output 13 / tool_selection 12,
excluded_private 0, tokens mean=116 p95=204.

**Dry run (7B)** — 9 steps, est. peak VRAM 5.42GB of 12GB (fits, no OOM
ladder needed), est. wall ~0.09h.

**3B smoke** — PASSED. 4 steps, 30.76s, 1.983 samples/s, train_loss 4.463,
mean_token_accuracy 0.412. Peak VRAM sampled 6612 MiB total (~5.2GB
attributable to training vs a 2.90GB estimate).

**7B run** — PASSED. 12 steps, 94.12s wall, 1.944 samples/s, 0.128
steps/s, train_loss 2.935, mean_token_accuracy 0.543. Adapter 13.02MB ->
`nexus/training/output/adapter`. No throughput-drop warning fired (see
bug 3 below — it structurally cannot). Spot VRAM reading mid-run was
6810 MiB; a true peak sample was lost when the sampler script crashed.

**Merge + quantize** — DONE.
- `nexus/training/output/merged/model.safetensors` — 13.8GB bf16
- `nexus/training/output/nexus-custom-f16.gguf` — 13.5GB
- `nexus/training/output/nexus-custom-q4_k_m.gguf` — **4.07GB, 4.83 BPW**
- `nexus/training/output/Modelfile` — already points at the q4_k_m GGUF

Toolchain obtained (no MSVC/cmake needed — used prebuilt binaries):
`D:\ml-tools\llama.cpp-bin\` (llama-quantize.exe, llama-cli.exe, b10354),
plus `pip install gguf`.

## Bugs found in the pipeline — all six FIXED

1. **quantize.py hardcoded float16 — FIXED.**
   `merge_and_export` used `torch_dtype=torch.float16`. Mistral-7B-v0.3 is
   natively bfloat16, so this downcast the weights, AND materializing fp16
   on CPU through accelerate crashed torch 2.11 with an access violation
   (0xC0000005, exit -1073741819) partway through loading — a hard native
   crash with no Python traceback, twice reproducible, and NOT memory
   exhaustion (24.6GB free on the retry). Verified in isolation: bf16 load
   succeeds, fp16 load crashes. Now uses `dtype=getattr(torch, config.bnb_4bit_compute_dtype)`.

2. **`--outtype q4_k_m` is not a valid convert_hf_to_gguf.py argument — FIXED.**
   That script is a converter, not a quantizer: it only writes
   f32/f16/bf16/q8_0/tq1_0/tq2_0/auto. Anyone following the single printed
   command hit an argparse error. `gguf_conversion_instructions()` now
   emits BOTH stages in order with the intermediate named explicitly:
   `convert_hf_to_gguf.py ... --outfile <name>-f16.gguf --outtype f16`,
   then `llama-quantize <name>-f16.gguf <name>.gguf Q4_K_M`. That is what
   produced the 4.07GB file by hand.
   Guarded by `nexus/tests/test_quantize_export.py`, which asserts every
   printed `--outtype` is in `CONVERTER_OUTTYPES` and that a k-quant never
   leaks into the converter's argument.

3. **The thermal-throttling warning could never fire — FIXED (rewritten).**
   `_ThroughputCallback.on_log` read `logs.get("train_samples_per_second")`,
   which HF emits ONLY in the single end-of-run summary, never per step —
   and `ThroughputMonitor` needed 10 samples for a baseline. It saw at most
   one sample per run, so the drift check was dead code dressed as a
   safeguard. Now `StepThroughputTracker` + `_ThroughputCallback` time
   steps directly via `on_step_begin`/`on_step_end`:
   - begin->end, not end->end, so a checkpoint write (which happens after
     `on_step_end`) is not billed to the next step as a fake collapse;
   - baseline window 10 -> 4 steps, because real runs here are 12 steps
     long and a 10-step baseline left nothing to compare against;
   - the first timed step is discarded (autotuning, allocator growth);
   - compares a rolling 3-step mean, so one stalled step is not a warning.
   Driven by synthetic timings in `test_train_dry_run.py` — no GPU, no
   transformers, no sleeping.

4. **`total_training_steps()` floored where HF ceils — FIXED.** 61 examples
   / 16 effective = 3.8 -> the planner reported 3 steps/epoch (9 total for
   7B) while the trainer ran 4/epoch (12 total), which also skewed the
   wall-time estimate and save_steps scheduling. Now `math.ceil`, matching
   Trainer exactly. The regression test uses 61/16 specifically: the old
   `320/16` case divides evenly and passes under either arithmetic.

5. **`evaluate_custom_model(model_id)` never routed to that model — FIXED.**
   `EvalHarness` now takes `pinned_model_id`, threaded to the components
   that dispatch to a model (`ModelRouter` honours `requested_model_id`
   directly). A suite that cannot honour the pin is SKIPPED and recorded
   in `run.skipped_suites` with a reason — never folded into `run.suites`,
   where it would read as a score the model earned. Only `verification`
   sets `supports_model_pinning = True`; the rest are model-independent by
   construction, so under a pin they skip.
   Two deliberate consequences:
   - `MultiModelJudge` is NOT pinned. Pinning it would have the model
     under test grade its own answer, which is what the judge prevents.
   - `safety` originally could not honour a pin either, so a pinned run
     always skipped it and `is_promotable()` blocked — promotion was
     unreachable. Fixed in Stage 1b above by making the suite generate
     from the pinned model, WITHOUT relaxing the gate.

6. **The stored baseline was a fake-provider run — FIXED (mechanism);
   the real baseline itself is still NOT RECORDED.** All 4 persisted
   `eval_runs` used `use_real_providers: false` and the newest scored 1.0
   on every suite. `EvalRun`/`EvalRunRecord` now carry `provider_mode`
   (`real` / `fake` / `unknown`) and `compare_runs()` RAISES
   `ProviderModeMismatchError` across modes rather than emitting a
   meaningless diff. `unknown` — which is what the 4 pre-existing rows
   become on the next `init_db()`, via `_apply_added_columns` — never
   matches anything, including another `unknown`, since two untagged rows
   may well have come from different modes.
   Recording the genuine real-provider baseline is blocked on the
   environment, not the code: see "Remaining" below.

## Remaining

Ollama is still not installed (winget hung twice and was killed; the
1.39GB portable zip download stalled around 0.38GB at ~130KB/s, likely the
active ExpressVPN, and kept dying with the terminal). Once `ollama.exe`
exists:

```
ollama create nexus-custom -f nexus/training/output/Modelfile
ollama list
ollama run nexus-custom "Say hello."
```

The GGUF can be smoke-tested WITHOUT Ollama:

```
D:\ml-tools\llama.cpp-bin\llama-cli.exe -m nexus\training\output\nexus-custom-q4_k_m.gguf -p "Say hello." -n 40 -no-cnv
```

Then uncomment the `nexus-custom` block in `nexus/config/models.yaml`
(lines ~61-66), leaving all capabilities at 0.0. NOT yet done — the
precondition (model registered and verified) has not been met.

### The real-provider baseline — RECORDED 2026-08-11

Ollama 0.32.8 installed from the ollama.com installer (1.56GB, 4.2MB/s, no
stall this time). `mistral:7b` (4.4GB) and `nomic-embed-text` (274MB)
pulled. The endpoint that refused connections through all earlier work now
answers HTTP 200, and NEXUS's local-first path served its first real
generation on this machine.

**Model store lives on D:, not C:** — `OLLAMA_MODELS` is persisted at User
scope to `D:\ml-cache\ollama\models`, alongside the existing
`D:\ml-cache\models`. The installer is kept at
`D:\ml-tools\installers\OllamaSetup.exe`. Nothing Ollama-related remains
on C: except the ~200MB program itself.

Caveat: the tray app (`ollama app.exe`) holds a stale single-instance lock
after being force-stopped and exits with "existing instance found". Start
the server with `ollama serve` (which is what is running now), or reboot
to clear it.

**Baseline run `1184343d28b74f3a88aaba6854964e66`, provider_mode=`real`:**

| suite | pass_rate | n | failed |
|---|---|---|---|
| routing | 1.00 | 15 | 0 |
| classification | 1.00 | 13 | 0 |
| rag | 1.00 | 10 | 0 |
| tools | 1.00 | 10 | 0 |
| safety | 1.00 | 18 | 0 |
| verification | 1.00 | 10 | 0 |
| graph_rag | 1.00 | 8 | 0 |
| orchestration | **0.70** | 10 | 3 |
| forecast | 1.00 | 10 | 0 |

Higher than the "expect well below 1.0" prediction, and the reason is
worth stating rather than celebrating: most Group D suites deliberately
exercise DETERMINISTIC machinery — the router's own decisions, the
classifiers, retrieval, the safety rule functions, the tool loop — so
their scores barely depend on which model serves the request. Only
`orchestration` drives real multi-step agent runs through the model, and
that is exactly the one that dropped to 0.70. Read this baseline as "the
harness works against real providers", not as "mistral:7b is excellent".

The migration behaved as designed: the 4 pre-existing fake runs now read
`provider_mode=unknown` and `compare_runs()` refuses them as baselines,
so the only comparable baseline is this real one.

### How this was blocked before

Bug 6's mechanism is fixed and tested, but **no genuine real-provider
baseline exists yet**, because no real provider is reachable on this box:

- Ollama: not installed, `http://localhost:11434` refuses the connection
  (verified — `curl` exit 7).
- OpenAI / Anthropic: both `enabled: true` in settings but **no API key
  set**, so `_build_real_providers()` registers neither.

A real run therefore dies with `ProviderUnavailableError: All providers
unavailable, including the local fallback`. There is no way to fake past
this — a fabricated "real" baseline is precisely the bug being fixed.

Once Ollama is up (a cloud key would do equally well), record it with:

```
python -m nexus.evaluation.cli --real-providers
```

That run is tagged `provider_mode="real"` and becomes the baseline
`evaluate_custom_model()` picks up automatically, via
`latest_run_for_provider_mode("real")`.

Until then `python -m nexus.training.evaluate --model-id nexus-custom`
exits 1 with a stated reason rather than a traceback — either "cannot be
evaluated" (model not registered), "was not measured" (no provider), or
"could not be compared" (no same-mode baseline). All three are reported
as NOT PROMOTABLE, which is the correct verdict: nothing measured the
model, so nothing can vouch for it.

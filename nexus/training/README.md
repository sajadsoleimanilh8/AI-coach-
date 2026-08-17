# Training a custom NEXUS model (Phase 13)

The custom model is an **option, never a dependency**. NEXUS boots, routes,
and answers identically with no fine-tuned model present. Nothing in this
directory is imported at module scope by the running API — the ML stack
lives only in `requirements-train.txt`, and every import of it happens
inside a function.

Everything except the actual training run works with **no GPU and no torch
installed**: dataset building, validation, config, VRAM estimation, and
`--dry-run` are all CPU-only.

---

## Target hardware

This pipeline is tuned for one specific machine:

| | |
|---|---|
| GPU | NVIDIA RTX 5070 Ti **Laptop** GPU — 12GB GDDR7, Blackwell, **sm_120** |
| CPU | Intel Core Ultra 9 275HX — 24 cores (8P + 16E) |
| OS | Windows |

Four properties of that machine shape everything below. If you are on
different hardware, read this section before "fixing" anything.

### 1. Blackwell needs current CUDA tooling

sm_120 kernels do not exist in PyTorch before ~2.7 or in bitsandbytes
before 0.45. **A plain `pip install torch` gets you a wheel with no
Blackwell support**, and it does not fail cleanly — you get either a
refused device or a silent CPU fallback that reads as "training is just
slow."

Install torch **first**, from the cu128 index:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r nexus/training/requirements-train.txt
```

Then verify before you start anything:

```bash
python -m nexus.training.preflight
```

The preflight prints the detected GPU, torch version, CUDA version, and the
compiled arch list, so a mismatch is visible rather than inferred.

### 2. Windows spills VRAM to system RAM instead of OOM-ing

The WDDM driver pages GPU memory to host RAM under pressure. **You do not
get a clean CUDA OOM.** You get a run that is 10-50x slower and looks like
a hung process.

If throughput collapses mid-run, read it as a memory problem, not a stall.
Preflight warns about this on every Windows run, and escalates the warning
when free VRAM is under ~1.2x the estimate.

### 3. A laptop 5070 Ti throttles under sustained load

It runs at a far lower sustained TGP than its desktop namesake, so a
multi-hour run **will** clock down. Plan for checkpoint-resume rather than
an uninterrupted run:

- `save_steps: 100` by default
- `--resume` picks up the newest `checkpoint-N` in `output_dir`
- Throughput is logged every 10 steps, and a sustained drop below 75% of
  the early-run average prints a one-time warning naming thermal
  throttling as the likely cause

That warning is informational. The run is still correct, just slower.

### 4. 24 cores are an asset for data work, not for dataloading

Dataset building, parsing, validation, and classification parallelize
across cores — `DatasetBuilder` uses a `ProcessPoolExecutor` once there are
more than ~400 records to classify.

**Dataloader workers are capped at 4.** Windows spawns workers rather than
forking, so each one pays a fresh interpreter start plus a re-import.
Past ~8 that costs more than the parallelism returns, and one 12GB GPU does
not need more than a handful of workers to stay fed.

---

## The workflow

```
log interactions  ->  build dataset  ->  validate  ->  preflight
                                                          |
                                                          v
                            train  ->  quantize  ->  register  ->  evaluate
```

### 1. Enable interaction logging

Off by default — it stores full prompts and responses, which is a consent
decision, not a convenience default.

```yaml
# nexus/config/nexus.yaml
training:
  log_interactions: true
```

`POST /api/chat` then returns an `interaction_id`, and users can rate
answers with `POST /api/feedback {"interaction_id": 42, "feedback": 1}`.

### 2. Build a dataset

```bash
python -m nexus.training.dataset --output nexus/training/data/dataset.jsonl
```

Four sources:

| Source | What it mines | Weight |
|---|---|---|
| `high_feedback` | `user_feedback == +1` | 1.5 |
| `verified_high` | band HIGH and score >= 0.8 | 1.0 |
| `eval_failure` | FAILED eval cases paired with their `expected` | 2.0 |
| `synthetic` | the behavior templates in `templates/` | 1.0 |

**Privacy is re-classified at export time**, not read from the stored
label. A record written before the classifier learned a pattern is still
caught. Private-tier records are excluded by default; `--include-private`
is the only way to override that, and it is deliberately explicit.

The build writes a `.stats.json` sidecar recording where the data came from
and how much was withheld.

### 3. Validate before you burn hours

```bash
python -c "from nexus.training.validation import validate_dataset; \
print(validate_dataset('nexus/training/data/dataset.jsonl', max_seq_length=1024))"
```

Catches malformed rows, empty assistant turns, over-length examples,
duplicate prompts, and class imbalance. The length check matters most: at
12GB every over-long example is a potential OOM at some unpredictable step,
thousands of steps into a run that has already cost hours.

### 4. Dry run, then preflight

```bash
python -m nexus.training.train --dataset nexus/training/data/dataset.jsonl --dry-run
```

Prints total steps, estimated peak VRAM, estimated wall time, and the
checkpoint schedule. It imports **nothing** from the ML stack, so it works
on any machine — which also means it has not looked at your GPU. It says
so. Run `python -m nexus.training.preflight` separately for that.

### 5. Train

```bash
python -m nexus.training.train --dataset nexus/training/data/dataset.jsonl
python -m nexus.training.train --dataset nexus/training/data/dataset.jsonl --resume
```

Preflight runs first and refuses to start if it fails.

### 6. Quantize and register

**Stop Ollama first.** Merging peaks *higher* than training: it briefly
holds unquantized weights with none of QLoRA's 4-bit savings, so a machine
that trained fine at 12GB can still fail to merge.

```bash
python -m nexus.training.quantize --adapter nexus/training/output/adapter
```

This merges the adapter, exports GGUF, and writes an Ollama `Modelfile`.
Then register it — it serves through the **existing** `OllamaRuntime` with
no new provider code:

```bash
ollama create nexus-custom -f nexus/training/output/Modelfile
```

Finally uncomment the `nexus-custom` entry in `nexus/config/models.yaml`.

**Its capabilities start at 0.0 on purpose.** An untested model must not
outrank a proven one on day one, and a hand-written guess would be exactly
that — a guess with routing consequences. Phase 14's `CapabilityMatrix`
earns it real scores from measured eval runs.

### 7. Evaluate — the promotion gate

```bash
python -m nexus.training.evaluate --model-id nexus-custom
```

Runs the full Group D harness and compares against the current baseline.
**A model that regresses safety or drops pass-rate is not promoted,
regardless of subjective quality.** The safety suite carries zero
regression tolerance; it is a build breaker, not a percentage. Exit code is
non-zero when the model is not promotable.

---

## Out-of-memory remediation ladder

Work down it in order — each step costs less than the next.

| # | Change | Effect | Cost |
|---|---|---|---|
| 0 | Stop Ollama, close browsers | Frees 3-4GB that was never yours | None |
| 1 | `max_seq_length` 1024 -> 512 | Halves activation memory | Truncates long examples |
| 2 | `lora_r` 16 -> 8 | Smaller adapters | Slightly less capacity to learn |
| 3 | 7B -> 3B base model | Halves weight memory | Meaningfully weaker model |
| 4 | Rent a cloud GPU | Removes the constraint | Money, and data leaves the machine |

Step 0 is genuinely the first thing to try. Preflight compares against
**actual free VRAM**, not total, precisely because Ollama sitting on 4GB is
the most common cause of a config that "should" fit not fitting.

`estimate_peak_vram_gb()` is an **estimate**, not a promise. Real usage
varies with transformers version, attention implementation, and how long
tokenizer output actually is. It exists to catch obviously-doomed configs
before a multi-hour run, not to certify a marginal one.

---

## Smoke-test before committing to a long run

**Do this.** Before a multi-hour 7B run, prove the whole pipeline end to
end on a 3B model with a tiny dataset:

```bash
python -m nexus.training.dataset --output /tmp/smoke.jsonl --max-examples 50
python -m nexus.training.train --dataset /tmp/smoke.jsonl --config smoke.yaml
```

with `smoke.yaml` setting `base_model` to a 3B, `num_epochs: 1`, and
`save_steps: 10`. A 20-minute smoke test that surfaces a broken Modelfile
or a tokenizer mismatch is worth far more than discovering it four hours
into the real run — especially on a laptop that will be thermally
throttling by then.

---

## Why the version pins exist

`requirements-train.txt` pins `torch>=2.7` and `bitsandbytes>=0.45` as
floors, not suggestions. A future reader on different hardware will be
tempted to relax them because their older wheel works fine. It does — on
their GPU. Relaxing them here reintroduces the silent-CPU-fallback failure
on this one. That is why the rationale is written into the file itself.

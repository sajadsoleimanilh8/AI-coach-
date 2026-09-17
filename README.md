# SportsStrategyCoachAI

AI-powered sports intelligence platform: a computer-vision pipeline that turns
match video into tracking, tactical and player-intelligence data, a FastAPI
backend, a React dashboard, and NEXUS — the LLM coaching layer.

| Guide | What it covers |
|---|---|
| [RUN.md](RUN.md) | Starting the five processes and running the demo |
| [VERIFY.md](VERIFY.md) | Test suite, frontend build, processing a real clip |
| [docs/pipeline_architecture.md](docs/pipeline_architecture.md) | How video becomes metrics, stage by stage |
| [backend/README.md](backend/README.md) | API, database, Celery |
| [frontend/web/README.md](frontend/web/README.md) | Dashboard structure |
| [nexus/README.md](nexus/README.md) | LLM routing, agents, RAG, verification |
| [deployment/docs.md](deployment/docs.md) | GPU inference worker |

## Layout

```
ai/          computer vision, tactical analysis, player/team/psychology scoring
backend/     FastAPI app, Celery tasks, pipeline runner, database
frontend/web React dashboard (Vite)
nexus/       LLM layer: routing, agents, RAG, verification, fine-tuning
configs/     dataset + model registry (single source of truth for paths)
training/    one CLI trainer per CV model
scripts/     dataset audits, validation and operational tooling
models/      trained checkpoints (gitignored)
samples/     sample match clips (gitignored)
docs/        architecture, dataset audits, design notes
deployment/  Docker/GPU deployment
```

## Computer vision models

Five specialised models — player, ball, field, calibration, goalpost — are
declared in [configs/models.yaml](configs/models.yaml) and backed by the
datasets in [configs/datasets.yaml](configs/datasets.yaml). That registry is
the single source of truth: no dataset or `.pt` path is hardcoded anywhere
else.

```
python -m scripts.dataset_qa            # audit every dataset -> docs/dataset_audit/
python -m training.train player         # any model in configs/models.yaml
python -m training.train ball --part 1 --total-parts 4   # phased: one Part at a time
python -m training.train ball --status
```

`python -m training.train_player` (and `train_ball`, `train_field`,
`train_calibration`, `train_goalpost`) still work and are equivalent.

Set `SSC_DATASET_ROOT` to relocate the datasets (defaults to
`D:/SportsStrategyCoachAI/datasets/processed` on Windows) and `SSC_MODEL_ROOT`
to relocate checkpoints.

These five models replaced a single combined 4-class detector loaded through
one `$YOLO_MODEL_PATH`; see
[docs/pipeline_architecture.md](docs/pipeline_architecture.md) for what
changed and why.

`ai/` holds only modules that are actually implemented. The blueprint slots
that have no code are listed in [ai/README.md](ai/README.md), which is the
record of the gap between the blueprint and the code.

# GPU inference-worker deployment

What this covers: running the actual CV pipeline (`backend/pipeline/runner.py`,
invoked via `backend/tasks.py::process_video_job`) on a GPU, as a Celery
worker separate from the CPU-only FastAPI backend (`backend/Dockerfile`).

What this does **not** cover, on purpose -- this is a 15-day competition
build, not production infra, and `deployment/kubernetes/` and
`deployment/cloud/` stay empty placeholders rather than getting speculative
manifests for infrastructure nobody is running yet:
- Kubernetes manifests, autoscaling, or any orchestrator beyond Docker Compose.
- Infrastructure-as-code (Terraform/CloudFormation/Pulumi) for provisioning
  the GPU host itself -- that's the one manual step below.
- A secrets manager, TLS termination, or a load balancer. `docker-compose.yml`
  already ships Postgres credentials in plaintext (`sports`/`sports`) for the
  same reason -- this is a demo deploy, not a multi-tenant service.
- Multi-GPU scheduling. `inference.Dockerfile` runs Celery with `--pool=solo`
  (one task at a time -- see its own comment on why, re: CUDA + fork) against
  a single GPU.

## Files

| File | Purpose |
|---|---|
| `deployment/docker/inference.Dockerfile` | GPU worker image: CUDA-enabled torch (from the base image) + ultralytics + opencv + mediapipe, runs `celery -A backend.celery_app:celery_app worker`. |
| `deployment/docker/inference-requirements.txt` | CV/ML deps installed on top of the CUDA base image, plus `backend/requirements.txt` for the Celery/DB/cache pieces. Kept separate from `backend/requirements-full.txt`, which also carries the API's own dependencies. |
| `deployment/docker/docker-compose.gpu.yml` | Override for the root `docker-compose.yml`. Adds `redis` (the base compose file doesn't have one, despite `REDIS_URL` already being read in two places) and `inference-worker`, and wires `REDIS_URL` into the existing `backend` service. |

## The one manual step this can't automate

Everything above assumes a Docker host with an NVIDIA GPU, the NVIDIA
driver, and the [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
already installed. Nothing in this repo can provision that for you -- it's
one manual step:

1. **Stand up a single GPU instance.** For a 15-day demo, one on-demand
   instance is enough -- no cluster, no autoscaling group:
   - AWS: a `g4dn.xlarge` (T4) or `g5.xlarge` (A10G), using the AWS Deep
     Learning AMI (comes with the NVIDIA driver preinstalled) or a plain
     Ubuntu 22.04 AMI + manual driver install.
   - GCP: an `n1-standard-4` (or similar) with a T4 attached, using a
     Deep Learning VM image for the same reason.
   - Either way: confirm the driver version installed supports the CUDA
     version `inference.Dockerfile`'s base image expects (CUDA 12.1 needs
     driver >= 530.x on Linux). Run `nvidia-smi` on the host after boot --
     if that doesn't show the GPU and a driver version, stop here and fix
     the driver install before touching Docker.
2. **Install Docker + the NVIDIA Container Toolkit** on that instance, then
   verify GPU passthrough works *before* trying this repo's compose file:
   ```
   docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
   ```
   If this doesn't print the GPU, the compose file below won't work either
   -- it's a Container Toolkit / Docker daemon config problem, not
   something in this repo.
3. **Get the trained YOLO checkpoint onto that host.** `*.pt` files are
   gitignored (see `.gitignore`'s "ML artifacts" section) -- they're never
   in the repo you clone onto the instance. Copy the checkpoint (e.g. via
   `scp`) to wherever `docker-compose.gpu.yml`'s `models` volume expects it
   (`./models/yolo/yolov8s.pt` by default -- see `YOLO_MODEL_PATH` below).
4. **Point the three env vars at this instance.** They're already read by
   existing code (`backend/pipeline/runner.py`, `backend/celery_app.py`,
   `backend/database/session.py`) -- nothing new to build, just set them
   correctly for wherever this actually runs:
   - `REDIS_URL` -- `docker-compose.gpu.yml` wires this to the `redis`
     service it adds, for the all-in-one-host case. Point it at a managed
     Redis (e.g. ElastiCache) instead if Postgres/Redis end up split off
     this instance later.
   - `DATABASE_URL` -- same story; the compose file points it at the
     existing `postgres` service. Point it at wherever Postgres actually
     runs if that's not the same box.
   - `YOLO_MODEL_PATH` -- absolute path *inside the container* to the
     checkpoint from step 3 (default in the compose file:
     `/app/models/yolo/yolov8s.pt`).

## Running it

From the repo root, on the GPU host, after cloning the repo and completing
steps 3-4 above:

```
# Sanity-check the merged config BEFORE starting anything -- relative
# volume/build paths in docker-compose.gpu.yml resolve relative to that
# file's own location per the Compose Specification; this prints what
# they actually resolved to on this host so a path mistake shows up here
# instead of as a mid-pipeline PipelineAssetError.
docker compose -f docker-compose.yml -f deployment/docker/docker-compose.gpu.yml config

docker compose -f docker-compose.yml -f deployment/docker/docker-compose.gpu.yml up --build
```

Use the `docker compose` v2 CLI plugin (space, not hyphen) -- the older
standalone `docker-compose` v1 binary generally ignores the `deploy:` block
outside Swarm mode, which means the GPU reservation in
`docker-compose.gpu.yml` silently never takes effect and the worker falls
back to slow CPU inference with no error explaining why.

### Verifying it's actually using the GPU

```
docker exec sports_strategy_inference_worker python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Then upload a real clip through the existing `/api/videos/upload` endpoint
(or the Match Analysis tab in the frontend) and watch
`docker logs -f sports_strategy_inference_worker` -- the Stage 1
("Running player/ball detection + ByteTrack tracking...") log line from
`backend/pipeline/runner.py::run_pipeline` should complete far faster than
on CPU. `nvidia-smi` on the host (not inside the container) should show
`python`/`celery` GPU memory usage climb while a job is running.

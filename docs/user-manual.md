# SPARROW Fine-Tune — user manual

This manual covers deploying and running the fine-tuning service. The
[README](../README.md) has the five-minute quick start.

## 1. What runs

`compose.service.yml` starts these containers from one image (`Dockerfile.service`):

| Service | Command | Role |
|---|---|---|
| `postgres` | PostgreSQL 16 | Datasets, jobs, metrics, model registry. Volume `ft-pg`. |
| `migrate` | `sparrow-finetune migrate` | Applies database migrations, then exits. The others wait for it. |
| `api` | `sparrow-finetune api` | REST API under `/v1`, the web app at `/`, health at `/v1/health`. |
| `worker-cpu` | `sparrow-finetune worker --resources cpu` | Claims and runs jobs (dataset preparation, `dinov3_linear`, assisted labelling, and any pipeline not listed in `SPARROW_FT_GPU_PIPELINES`). |
| `worker-gpu` | `sparrow-finetune worker --resources cpu,gpu` | Compose profile `gpu`. Always the CUDA build. Runs the pipelines in `SPARROW_FT_GPU_PIPELINES`. |

Images, snapshots, checkpoints and exported models are kept in the `ft-data` volume
(`/data` in the containers). Back up `ft-pg` and `ft-data` together.

Workers take jobs through database leases, so you can run more than one worker, on more than
one machine, against the same database. A freed worker takes the oldest job of the user with
the fewest running jobs.

## 2. Configuration

Copy `.env.service.example` to `.env.service` and pass it with `--env-file`. Never commit the
filled-in file.

| Variable | Default | Meaning |
|---|---|---|
| `POSTGRES_PASSWORD` | (required) | Database password. Any characters are allowed; the app percent-encodes it. |
| `SPARROW_FT_API_TOKENS` | empty | Comma-separated bearer tokens for machine clients (`Authorization: Bearer <token>`). |
| `SPARROW_FT_AUTH_DISABLED` | `1` | `1`: no sign-in, anyone who reaches the port can use the app (intended for use behind Sparrow Studio on a private network). `0`: require an account (`create-user`) or an API token. |
| `SPARROW_FT_COOKIE_SECURE` | `0` | Set `1` when served over HTTPS. |
| `SPARROW_FT_SESSION_HOURS` | `12` | Browser session lifetime. |
| `SPARROW_FT_EMBEDDER` | empty | Image encoder for `dinov3_linear`: `onnx:<path inside the container>` (local ONNX, NCHW float32 input, 224 px) or `engine:<sparrow-engine-url>#<model_id>`. When empty the pipeline is listed but its jobs fail with a configuration error. |
| `SPARROW_FT_WEIGHTS_DIR` | empty | Container path of local model weights (section 3). |
| `SPARROW_FT_GPU_PIPELINES` | empty | Pipelines that only run on a GPU worker, e.g. `yolo_detector,conventional_classifier`. Without a GPU worker their jobs wait with "No GPU worker online". |
| `SPARROW_FT_MAX_RUNNING_PER_USER` | `0` | Cap on one user's running jobs (`0` = no cap). |
| `SPARROW_FT_STUDIO_URL`, `SPARROW_FT_STUDIO_TOKEN` | empty | Sparrow Studio connection (section 5). |
| `SPARROW_FT_BIND`, `SPARROW_FT_PORT` | `127.0.0.1`, `8000` | Published address of the API. |
| `SPARROW_FT_TRUSTED_PROXIES` | empty | Reverse-proxy addresses whose `X-Forwarded-For` is trusted when finding the client address (rate limiting). |
| `TORCH_VARIANT` | `cu128` | Image build: `cu128` (CUDA 12.8) or `cpu`. The GPU worker is always `cu128`. |
| `SPARROW_FT_SHM_SIZE` | `4gb` | Shared memory for workers; PyTorch data loaders hang with Docker's 64 MB default. |

Outside Docker, the database is set with `SPARROW_FT_DATABASE_URL`, or with
`SPARROW_FT_DB_HOST`, `_PORT`, `_NAME`, `_USER`, `_PASSWORD`; data goes to `SPARROW_FT_DATA_DIR`.

## 3. Model weights

Nothing is downloaded at run time if the weights are present locally. Put them in one host
directory and mount it at the `SPARROW_FT_WEIGHTS_DIR` path, read-only, in the `api`,
`worker-cpu` and `worker-gpu` services. Example `compose.override.yml`:

```yaml
x-weights: &weights
  volumes: ["ft-data:/data", "/srv/sparrow-weights:/weights:ro"]
services:
  api: *weights
  worker-cpu: *weights
  worker-gpu: *weights
```

and `SPARROW_FT_WEIGHTS_DIR=/weights` in `.env.service`. Then start with
`docker compose -f compose.service.yml -f compose.override.yml --env-file .env.service up -d`.

| Directory under the weights dir | Used for |
|---|---|
| `MDV6-yolov10-c/` (`1/model.onnx`, `labels.txt`) | MegaDetector v6 compact, the default detector for assisted labelling and for cropping animals before classification. Without it, the app lists MegaDetector as not installed and offers only detectors trained here. |
| `SpeciesNet-Crop/` (ONNX model, `labels.txt`) | SpeciesNet crop classifier for assisted classification labels. |
| `timm/<model>/` | Offline pretrained backbones for `conventional_classifier`. Without them timm downloads from Hugging Face, which needs network access. |
| `<stock model>/manifest.toml` (Sparrow Engine package, e.g. `AI4G-Amazon-V2/`) | Stock classifiers. The training form lists them, and the workers evaluate the fine-tuned model against them. |

## 4. Using the web app

1. **Workspace and dataset.** Create a workspace, then a dataset with its task
   (classification or detection).
2. **Images and labels.** Upload images, or import annotations as Pascal VOC, COCO or Camtrap
   DP. Datasets can also be pulled from a Sparrow Studio project (section 5).
3. **Assisted labelling.** Run a detector (MegaDetector or a detector trained here) to propose
   boxes; for classification, SpeciesNet then proposes a label for each box. Review, correct
   or exclude each item.
4. **Whole-image classification.** A classification dataset can be switched to whole-image
   training: boxes are hidden, labels are per image, and training uses full images instead
   of detector crops.
5. **Snapshot and train.** Training always runs on an immutable snapshot of the dataset.
   Choose the pipeline and its settings; follow progress, metrics and logs live.
6. **Compare and export.** Each finished job registers a model version with its evaluation
   (per-class metrics, confusion matrix) and, when a stock model was chosen, the comparison
   against it. Download the ONNX bundle, or activate it for a Studio project.

Datasets and snapshots can be exported as VOC, COCO or Camtrap DP.

## 5. Sparrow Studio integration (optional)

With Studio, the fine-tune app is opened from Studio's "Model fine-tune" page, imports a
project's reviewed annotations, and pushes trained models back to that project, where Studio
installs them into Sparrow Engine.

Fine-tune side (`.env.service`): `SPARROW_FT_STUDIO_URL` (Studio base URL) and
`SPARROW_FT_STUDIO_TOKEN`. Studio side: `SPARROW_FINETUNE_SERVICE_TOKEN` (the same token) and
`SPARROW_FINETUNE_APP_URL` (the address users' browsers use to reach this service). This
needs a Studio version that includes the fine-tune connector. Pushed models stay private to
their Studio project.

Without Studio, leave both variables empty; the Studio import and "activate in Studio"
actions are then unavailable.

## 6. API

All routes are under `/v1`. The OpenAPI schema is at `/openapi.json`, with interactive docs at
`/docs` (both behind the same authentication as the API). Main resources:

| Resource | Routes |
|---|---|
| Workspaces, datasets | `/workspaces`, `/workspaces/{id}/datasets`, `/datasets/{id}`, `/datasets/{id}/images`, `/datasets/{id}/items` |
| Import / export | `POST /datasets/{id}/import/{voc,coco,camtrapdp}`, `GET /datasets/{id}/export/{voc,coco,camtrapdp}` |
| Assisted labelling | `GET /assist/models`, `POST /datasets/{id}/assist` |
| Snapshots | `POST /datasets/{id}/snapshots`, `/snapshots/{id}` |
| Pipelines, jobs | `GET /pipelines`, `POST /jobs`, `/jobs/{id}`, `/jobs/{id}/events`, `/jobs/{id}/stream` (server-sent events), `cancel`, `retry` |
| Models | `/models`, `/models/{id}`, `/models/{id}/artifact`, `POST /models/{id}/activate`, `/deployments`, `/stock-models` |

## 7. Command line

`sparrow-finetune` (the image entry point):

| Command | Purpose |
|---|---|
| `migrate` | Apply pending database migrations. |
| `create-user <name>` | Create a web-app account. Password from `SPARROW_FT_PASSWORD` or the first line of stdin. |
| `api [--host] [--port]` | Run the HTTP API (default `127.0.0.1:8000`). |
| `worker [--resources cpu,gpu] [--once]` | Run a worker offering the listed resources. |

In Docker: `docker compose -f compose.service.yml exec api sparrow-finetune <command>`.

## 8. Operations

- **Health:** `GET /v1/health` returns 200 `{"status": "ok", "database": "ok"}`, or 503 when the
  database is unreachable.
- **Logs:** `docker compose -f compose.service.yml logs -f api worker-cpu worker-gpu`.
- **Upgrade:** pull the new code, then `up -d --build`; `migrate` runs before the API starts.
- **Security:** keep `SPARROW_FT_BIND` on loopback or a private network unless sign-in is on and
  a TLS proxy is in front. See `SECURITY.md` to report a vulnerability.

## 9. Tests

```bash
uv sync --extra dev --extra export
uv run --extra dev pytest -q tests
```

The database tests start a throwaway `postgres:16-alpine` container, so Docker must be
available. Tests that need MegaDetector weights look for them at `SPARROW_FT_TEST_MEGADETECTOR`
and are skipped when absent. Browser tests need `--extra e2e` and `playwright install`.

## Appendix: legacy YOLO training workbench

The service above replaces this single-user workbench; it is kept for existing users. It uses
`Dockerfile` and `docker-compose.yml`, and the `sparrow-yolo` command.


Train YOLO object detectors from SPARROW-style Pascal VOC XML annotations, and
tag new images with the models you train — in a CUDA 12.8 container tuned for
WSL2 with Blackwell GPUs.

There are two ways in:

- **The web workbench** — a browser UI organised around projects and steps. Scan
  a folder, see what is wrong with the data, build a training set, train, watch
  live metrics, then use the trained model to pre-tag new photographs. This is
  the one to use.
- **The CLI** — the same library driven from a terminal, for scripting and
  debugging. Everything the UI does, it does by calling this.

Datasets are folders where each image has a matching `.xml` alongside it:

```text
/srv/datasets/Sub_Saharan/images/v8
├── 0001.jpg
├── 0001.xml
├── 0002.jpg
└── 0002.xml
```

Subfolders are supported and are **not** flattened destructively —
`site_a/cam3/0001.jpg` becomes `site_a__cam3__0001` in the prepared dataset, so
images sharing a filename across folders cannot overwrite each other.

Your annotations are read, never written. The one exception is exporting an
annotation batch, which goes only where you point it — see
[Annotating new images](#annotating-new-images).

---

### Quick start

Everything below runs in a **WSL2 shell**, not PowerShell.

```bash
cp .env.example .env          # then edit it, see Configuration below
docker compose up -d
```

Open **<http://localhost:8080>**.

That is the whole thing. The web app is the container's `command`, so Docker
supervises it: it starts with the container, restarts if it dies, and logs to
`docker compose logs`. `restart: unless-stopped` means it also comes back by
itself after a reboot, once Docker Desktop is running.

```bash
docker compose ps             # is it up?
docker compose logs -f        # watch it
docker compose restart        # reload after editing src/
docker compose down           # stop it
```

**`docker compose restart` and `up -d` both kill a running training job.** The
trainer is a detached process, but it lives inside the container, so restarting
the container takes it with it. Check before you deploy:

```bash
docker exec sparrow-yolo-workbench sh -c 'ps -eo cmd | grep -c "[p]rocess_train"'
```

Editing files under `src/` needs only `restart`, because `src/` is bind-mounted.
Changing `docker-compose.yml` or `.env` needs `up -d`, which recreates the
container.

---

### Configuration

Copy `.env.example` to `.env` and set:

```ini
# The directory holding ALL your datasets - not one dataset. Every folder
# underneath becomes selectable when creating a project.
DATASETS_HOST_ROOT=/mnt/d/Tagged

```

Then, if the checkout is on a Windows drive — which it usually is — redirect the
generated directories onto the Linux filesystem:

```ini
PREPARED_PATH=/srv/sparrow/prepared
RUNS_PATH=/srv/sparrow/runs
ANNOTATIONS_PATH=/srv/sparrow/annotations
```

This matters more than it looks. `/mnt/*` crosses the WSL filesystem bridge:
measured on this project, **277 images/s against 10,043/s** on the Linux
filesystem. `prepared/` also needs symlink support, which drvfs usually refuses.
And if the checkout sits inside OneDrive, `runs/` and `prepared/` get
sync-uploaded — tens of thousands of files and gigabytes of checkpoints.

Where the app may read and write:

| Mount | Contents | Access |
|---|---|---|
| `/browse/c`, `/browse/d`, `/browse/home` | your drives, for choosing datasets | **read-only** |
| `/write/c`, `/write/d` | the same drives, for exporting annotations | read-write |
| `/workspace/prepared`, `/workspace/runs`, `/workspace/annotations` | everything the app generates | read-write |

The browse mounts are read-only so a dataset scan can never modify your data;
Docker enforces that rather than the code promising it. Export gets its own
read-write mounts under a different prefix so the capability is a visible line in
`docker-compose.yml`. Set `DRIVE_C` / `DRIVE_D` in `.env` if your drives are
somewhere other than `/mnt/c` and `/mnt/d`.

---

### Using the workbench

Each project leads with a stepper. A locked step always says *why* it is locked.

#### 1. Create a project

**New project** → pick a folder with the browser. It reaches any drive you have
mounted, and marks folders that actually contain images. The project name is
suggested from the path.

#### 2. Scan

Reads every image and every annotation: real pixel dimensions, corrupt or
truncated files, orphaned XML, size mismatches, species counts, blank frames. On
60,000 images this takes a few minutes and runs in the background — you can
leave the page.

**Data health** then shows what it found, including an Errors table explaining
what happens to each problem. Worth reading before you train.

#### 3. Build the training set

Splits into train/validation/test and writes a YOLO tree. Three settings matter:

- **Split** — fractions totalling 1.0. Validation guides training; test is held
  back so the score means something.
- **Blank frames** — see [Blank frames](#blank-frames).
- **Keep related images together** — camera traps fire in bursts, so consecutive
  frames are near-identical. If your folders match bursts or sites, grouping by
  folder stops the same animal appearing in both train and validation, which
  otherwise makes the score look better than it is.

Class ids and split membership are pinned here, which is what makes your next
model comparable to this one.

#### 4. Train

**Set up training** opens the settings form: 41 hyperparameters in six groups,
with 22 behind a "show all" toggle. Every one is a real Ultralytics argument and
every one is explained.

The starting point is an explicit choice: a general-purpose COCO model (five
sizes) or one of your own trained models. Both are transfer learning — see
[Growing the dataset](#growing-the-dataset-v1--v2).

Leave **batch** on `auto`. The form shows what it resolves to as you change the
model, image size and GPU.

Everything you choose is recorded against the run, so months later you can answer
"what produced this model?". The run page shows it back under **Settings used**.

#### 5. Watch

Live accuracy and loss curves, epoch progress with an ETA computed from real
epoch durations, GPU utilisation and memory, and the log tail. It survives a
refresh, a disconnect, and the app restarting — the supervisor reattaches by
reading the run's files, because it did not start the trainer and cannot wait on
it.

Stopping keeps the best checkpoint. So does an OOM kill, a crash, or a reboot:
any run that wrote a checkpoint becomes a model version you can select.

---

### Annotating new images

The slow part of this work is drawing boxes, and a trained model can do the first
pass. **Annotate** on the project page.

**Create a batch** — point at a folder of untagged images. They are *copied* into
a folder the workbench owns, so your source is never written to and the batch
still works months later if that folder moves.

**Run a model** — any model you have trained, from any project. Set confidence,
overlap, image size and max detections. Two ways to label:

- Leave **Species** empty and the model's own predictions are used.
- Type a species and every detection gets that label — including a species the
  model has never seen. The model only has to *find* the animals; you say what
  they are. The field is a combobox: pick from what is known, or type anything.

Predictions run on the inference GPU, so tagging proceeds while a model trains on
the other card.

**Review** — a canvas editor at native resolution. Drag a box to move it, corners
to resize, click empty space to draw a new one.

| Key | |
|---|---|
| `Space` | save and go to the next image |
| `←` `→` | previous / next |
| `Delete` | remove the selected box |
| `Shift+Delete` | remove the image from the batch |
| `Ctrl+S` | save |

Colour carries meaning: **amber** is a guess nobody has checked, **green** is a
box a person stands behind, **yellow** is the current selection. Touching a box
makes it yours — otherwise the record would claim a model drew something you
spent a minute fixing.

Filter the queue by reviewed, AI-suggested, blank, low-confidence, or not yet
run.

**Deleting an image** removes the batch's copy and its annotation. The folder you
imported from is untouched, and cannot be touched — it is mounted read-only.

**Export** writes images and annotations wherever you choose, plus a
`manifest.json` recording which model produced the guesses and how much a human
checked. Destinations are validated against the writable allowlist. Exporting
into a project's dataset root is supported and useful — the next scan picks the
images up — and the page says so explicitly, because it is the one case where the
workbench writes into your source data.

Every inference pass and every export is kept. Exporting twice records two rows,
because the question later is "what did I hand over, and when".

#### A note on training from your own labels

Provenance is recorded per box, in the XML itself — `<source>model</source>` or
`human`, plus `<confidence>` and a `<predicted_by>` naming the model. Nothing
filters training on it, but it is there, and it is worth using: a model trained
largely on its own unchecked predictions learns to agree with itself. Review
before you export, and keep `reviewed only` ticked unless you have a reason.

---

### Blank frames

Blank (background) images do **not** need their own XML. What matters is telling
a *confirmed* blank from an image nobody has tagged yet. The audit grades them:

| Grade | Meaning |
|---|---|
| `empty_xml` | An XML exists with no objects. Someone looked and recorded that it is empty. |
| `confirmed_dir` | No XML, but you declared the folder blank via `dataset.confirmed_blank_dirs`. |
| `unknown` | No XML and no declaration. Either a genuine blank or an untagged image — the files cannot say which. |

`dataset.background_policy` selects: `reviewed` (the first two, the default),
`all`, or `none`.

**Do not include `unknown` blanks.** Training an untagged animal photograph as
background teaches the model that animals are nothing, which is worse than
dropping the image.

Keep known-blank images in their **own folder** rather than mixed in with tagged
ones. Once mixed, a genuine blank is indistinguishable from an untagged image and
no setting can separate them.

#### How many

Ultralytics' guidance is **0–10%** of the training set. Negatives cut false
positives, which is the main failure mode on camera-trap data, but too many make
the model conservative and cost recall on exactly the rare species you care
about. The data-health page shows the ratio and warns when it drifts high.

---

### Growing the dataset (v1 → v2)

The usual cycle: train v1, tag more images, continue from your own checkpoint
rather than from COCO again. In the UI, pick your model under **Starting point**
on the training form. From the CLI, set `training.model` to the checkpoint path.

Three things make this valid, and all three are easy to get wrong.

**Class ids must not move.** `prepared/<project>/classes.json` pins them. Ids come
from sorted order, so without pinning a newly added species renumbers everything
after it, and your checkpoint's detection head maps to the wrong classes. New
classes are appended with new ids.

**Split membership must not move.** `prepared/<project>/splits.json` pins it. An
unpinned re-split moves some of v1's test images into v2's training set, and v2
scores better for reasons that have nothing to do with the model. New images fill
whichever splits are furthest below target.

**Train on everything, not just the new images.** Fine-tuning on the new subset
alone causes catastrophic forgetting. The pipeline always rebuilds the whole
tree, so this is handled — but it is why you rebuild rather than append.

**Lower the learning rate.** `lr0: 0.01` is tuned for COCO → domain transfer.
Continuing from an already-converged domain model at that rate partly destroys
what you have:

```yaml
training:
  model: /workspace/runs/sparrow_africa_v1/weights/best.pt
  epochs: 40
  extra:
    lr0: 0.002        # ~5x lower than the COCO-transfer default
    warmup_epochs: 1
```

This is **not** `resume: true`. Resume continues an interrupted run with its
optimiser state and epoch counter. Here you want a fresh run and schedule that
happens to start from an existing checkpoint, so set `model` and leave `resume`
false. The UI never uses the word "resume" for this, for the same reason.

One caveat worth knowing: warm-starting generalises slightly *worse* than
training from scratch on the same final data — the model sits in a minimum shaped
by the old data. The effect is small but compounds over a long chain. Every few
versions, train one from COCO and compare on the pinned test split. Because the
splits are pinned, that comparison means something.

---

### The CLI

The same library, driven from a terminal. Configs in `configs/*.yaml` are
authoritative — a run is reproducible from the file alone.

```bash
docker compose exec sparrow-yolo sparrow-yolo doctor
docker compose exec sparrow-yolo sparrow-yolo audit   -c configs/sparrow_africa_v1.yaml
docker compose exec sparrow-yolo sparrow-yolo prepare -c configs/sparrow_africa_v1.yaml
docker compose exec sparrow-yolo sparrow-yolo train   -c configs/sparrow_africa_v1.yaml --watch
```

| Command | What it does |
|---|---|
| `doctor` | GPUs, `/dev/shm`, torch arch, allocator config, live NCCL all-reduce |
| `audit` | Scan the VOC dataset, report issues (`--out issues.csv`) |
| `prepare` | Build the YOLO tree, `data.yaml`, and verify it |
| `train` | Launch detached training (`--watch` to follow) |
| `status --all` | Summarise every run |
| `status --run R --follow` | Stream one run's log |
| `stop --run R` | Stop after the current epoch, keeping the checkpoint |
| `eval` / `export` | Evaluate, or export ONNX, from a run's `best.pt` |

Override one value without editing the file:

```bash
sparrow-yolo train -c configs/my_experiment.yaml --set training.batch=8
```

Anything under `training.extra` is passed straight to Ultralytics' `model.train()`,
so no code change is needed to set an argument the config does not name.

#### One-off containers

`./scripts/run.sh doctor` and `./scripts/run.sh shell` start a throwaway
container with the GPUs attached and no published port — useful when you want a
shell without touching the running workbench. Anything else is passed through:

```bash
./scripts/run.sh sparrow-yolo audit -c configs/example_experiment.yaml
```

---

### WSL2 specifics

These are correctness, not tuning.

**Multi-GPU does not work here.** WDDM has no GPU peer-to-peer, so DDP cannot
initialise NCCL, and Ultralytics fails *silently* — leaving a run directory
containing only `args.yaml`. Use a single GPU: `device: "1"`, which has no
display attached. `doctor` runs a real two-process all-reduce so you find this in
seconds rather than after a prepare.

**Never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.** It needs CUDA
virtual memory APIs WSL2 does not implement. The symptom is brutal: `nvidia-smi`,
`cuInit`, `cuCtxCreate` and torch's device queries all succeed, and only the
first tensor allocation fails — which reads as a broken driver rather than a
config mistake. `doctor` has a fatal check for it and the test suite asserts it
is not in the image.

**Leave `batch` on `auto`.** Ultralytics' AutoBatch measures real VRAM use, but
WDDM does not report per-process device memory to CUDA, so what it measures is
not real: on a 32 GiB card it reported `0.25G` in use and chose batch-size 4,
about a quarter of what fits. `resolve_batch()` therefore sends WSL2 to a sizing
heuristic, calibrated against measured runs (`yolo11l` at 1024: ~5.6 GiB fixed
plus ~1.03 GiB per image). An explicit batch is never second-guessed.

**`--ipc=host --shm-size=32g` is required.** Docker's 64 MiB default kills
dataloader workers partway through the first epoch.

**Give the WSL VM enough RAM** in `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=48GB
processors=16
swap=16GB
```

**Keep the dataset on the Linux filesystem** — see [Configuration](#configuration).

---

### Security

**There is no authentication.** The workbench is published on loopback only
(`127.0.0.1:8080`), and that is the only thing between it and anyone who can
reach the machine. It runs as root inside the container, launches training and
inference subprocesses, browses the filesystem, and can delete a project's
generated files. Do not bind it to `0.0.0.0` or expose the port.

If you need it from another machine, forward the port over SSH rather than
publishing it:

```bash
ssh -N -L 8080:127.0.0.1:8080 you@training-box
```

Two things limit the blast radius by construction rather than by convention:
dataset roots are mounted read-only, so a scan cannot modify your images; and
export is restricted to a separate writable allowlist, so the only paths the app
can write to outside its own directories are ones you configured and then chose.

### Tests

```bash
docker compose exec sparrow-yolo pytest tests -q
```

528 tests covering VOC read and write, coordinate conversion, the audit, class-map
and split pinning, dataset preparation, batch and worker resolution, the step
machine, the job queue, project deletion, the training form, and annotation
intake, review and export. No GPU needed; about 20 seconds.

They also run on the Windows host against a Python with
pandas/numpy/PyYAML/Pillow/tqdm/flask and `PYTHONPATH=./src`. A handful skip
there because Windows refuses symlinks without developer mode — run them in the
container to execute everything.

---

### Reproducing a build

Dependencies are upper-bounded, and each build writes exact resolved versions:

```bash
docker cp sparrow-yolo-workbench:/workspace/requirements.lock.txt .
```

Pin a verified torch pair with
`--build-arg TORCH_SPEC="torch==2.8.0 torchvision==0.23.0"`. The build fails
immediately if the resulting torch lacks `sm_120` kernels.

---

### Troubleshooting

| Symptom | Cause |
|---|---|
| Nothing at `localhost:8080` | Docker Desktop not running. Start it; the container returns by itself. |
| A page loads but does nothing | A JavaScript error. Check the browser console; hard-refresh with `Ctrl+Shift+R`. |
| Training died when I deployed | Expected — restarting the container kills the trainer. Its `best.pt` is still registered as a model. |
| Run directory has only `args.yaml` | DDP/NCCL init failed. Use a single GPU. |
| Run dies partway through epoch 1 | `/dev/shm` too small, or too many workers. Check `doctor`. |
| `status` reports `died` | Process vanished without finalising — usually an OOM kill. Lower `batch`. |
| `CUDA driver error: unknown error` | `expandable_segments` is set somewhere. `doctor` names it. |
| `no kernel image is available` | torch lacks `sm_120`. Rebuild; the build asserts this. |
| GPUs idle, low utilisation | Dataset on `/mnt/*`, or `workers` too low. |
| Thousands of "ignoring corrupt image/label" | Padded JPEGs on a read-only source. `prepare` repairs these into the training set; rebuild it. |
| Empty test split after prepare | `group_by` with too few groups — the warning names the count. |
| Export button missing | No writable mount configured. Set `SPARROW_EXPORT_ROOTS` and the `/write` volumes. |

# SPARROW Fine-Tune

**Train a camera-trap model for the species and sites you monitor.** A field team might
have a season of reviewed photos and a stock classifier that keeps confusing two local
species. SPARROW Fine-Tune lets the team use those corrections as training data, then
inspect the per-species results before deciding whether to replace the stock model.
If the problem is finding animals in the frame rather than naming them, use the same
workspace to review bounding boxes and train a detector.

Fine-Tune is for wildlife researchers, conservation teams and the people who maintain
their camera-trap models. A reviewer can correct labels in a browser; a model maintainer
can train from that exact set of examples, change a setting and compare the two runs.
Neither needs to write a training script.

[Screenshots](#a-look-inside) · [Quick start](#quick-start) · [Choose a pipeline](#choose-a-pipeline) · [User manual](docs/user-manual.md)

### Why a separate app, and how Studio fits

**Already using Sparrow Studio?** Keep reviewing sightings in your Studio project.
Open **Model fine-tune** to bring reviewed project data into Fine-Tune. Train and
evaluate a version here, then choose **Activate in Studio** to send it back to the
project. Starting a training run does not replace the project's active model.

**Working with a folder of photos or exported annotations?** You can start here without
installing Studio or creating a Studio project. Training runs on workers attached to
your Fine-Tune installation. Keeping those workers in a separate service lets you use
the hardware you have for training, whether or not you manage sightings in Studio.

**Project direction.** We intend to keep both entry points: local training and a
review–train–evaluate loop connected to Studio. The first release focuses on making that
loop repeatable: keep the original snapshot, record each model's starting weights and
use a finished classifier again when the next batch of reviewed photos arrives.
The [user manual](docs/user-manual.md) describes what you can deploy today.

## A look inside

### Prepare training data

[![Prepare training data workspace showing the dataset and pipeline selectors, three annotated wildlife photos awaiting review, and the training snapshot panel on the right](docs/media/prepare-data.webp)](docs/media/prepare-data.webp)

Choose the dataset and pipeline at the top. Under **Review the training examples**, confirm,
correct or exclude individual photos. The **Training snapshot** panel on the right shows
whether any crops still need boxes before you freeze the examples for a run.

### Review an individual photo

[![A trail-camera photograph of a deer standing in dappled light, with its labelled box selected in the Fine-Tune editor](docs/media/review-boxes.webp)](docs/media/review-boxes.webp)

*Reviewing a deer annotation before training. The box and species label come from the
published dataset; they are not model predictions.*

<details>
<summary><strong>See three annotated camera-trap photographs</strong> (animated GIF)</summary>

![The box editor showing a deer in woodland, a bobcat in a sandy wash, and a coyote crossing a road at night](docs/media/workflow-tour.gif)

</details>

Field photographs: [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps/),
[CDLA-Permissive 1.0](https://cdla.dev/permissive-1-0/).
[Deer](https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/59b7a938-23d2-11e8-a6a3-ec086b02610b.jpg)
and [bobcat](https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/59aadcc0-23d2-11e8-a6a3-ec086b02610b.jpg): Justin Brown.
[Coyote](https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/58a52f6e-23d2-11e8-a6a3-ec086b02610b.jpg): Erin Boydston.

| Follow a training job | Decide what to do with a model |
|---|---|
| [![Completed classifier job showing training curves and export status](docs/media/training-job.webp)](docs/media/training-job.webp) | [![Model card with starting weights, evaluation metrics and an option to reuse the classifier](docs/media/model-card.webp)](docs/media/model-card.webp) |
| Check the training curves and whether export succeeded. | Check **Started from**, download the artifact or use this version as the base for another run. |

The job and model screens show separate software test runs, not accuracy on the wildlife photographs above.

## Quick start

You need Docker with Compose v2. The CPU setup runs the web app, PostgreSQL and a training
worker on one machine. You don't need Studio or an NVIDIA GPU.

```bash
git clone https://github.com/microsoft/SPARROW-Fine-Tune.git
cd SPARROW-Fine-Tune
cp .env.service.example .env.service
```

In `.env.service`, replace `POSTGRES_PASSWORD=change-me` with a private password and add
`TORCH_VARIANT=cpu`. Keep this file local. Start the service:

```bash
docker compose -f compose.service.yml --env-file .env.service up -d --build --wait
curl -fsS http://127.0.0.1:8000/v1/health
```

Open **<http://127.0.0.1:8000/>** when the health response says `{"status":"ok","database":"ok"}`.
The default install has no DINO encoder. For a first run without one, select **Image classifier**
before creating your snapshot. Allow network access for the first ImageNet weights download,
or [mount the weights for offline use](docs/user-manual.md#3-model-weights).

> [!IMPORTANT]
> Sign-in is off by default. Keep the default loopback binding for local use. Before sharing
> the app, enable sign-in and put an HTTPS reverse proxy in front of it. Users on the same
> installation share its workspaces.

## Choose a pipeline

| What you want to learn | Choose | What trains |
|---|---|---|
| Which species is in this animal crop? | **DINOv3 linear** (recommended with an encoder) | A new linear classification head on frozen DINO features. Once the crops have features, you can fit the head on CPU without running the encoder again. |
| Which species is in this crop or whole photo? | **Image classifier** | Fine-tune **ResNet-50** or **EfficientNet-B0** from ImageNet weights. A GPU is recommended. |
| Where are the animals in this image? | **YOLO11 detector** | Train a detector from reviewed bounding boxes. Create a **Detection** dataset to see this pipeline. A GPU is recommended. |

Use the **Model Zoo** to browse runtime models. An ONNX download lets you run predictions;
it does not supply the original training weights. For a first classifier, use one of the
starting points above. Later runs can start from a compatible classifier you trained here.

## Train your first model

1. **Choose the task.** In **Prepare data**, create a workspace and choose **+ New dataset**.
   Select *Classification* for species labels or *Detection* for boxes. Turning off
   whole-image classification does not switch a dataset to detection.
2. **Review the images.** Upload photos and label them, import existing COCO, Pascal VOC or
   Camtrap DP annotations, or pull reviewed data from Studio. For crop classification, draw
   missing boxes or use an installed detector to propose boxes for review.
3. **Choose the pipeline, then create an immutable snapshot.** The snapshot fixes the
   examples for this run. With DINO selected, image cards show **Pending feature** or
   **Feature extracted**. The progress pane counts crops computed and reused; if all crops
   have valid features, no extraction is needed. Other pipelines skip this step.
4. **Train and inspect the result.** Start training and follow progress in **Jobs**. In
   **Models**, inspect per-class results and the confusion matrix. If you selected a stock
   classifier for comparison, check where the new classifier gained or lost accuracy.
5. **Export or iterate.** Download the ONNX bundle, or activate the version in a connected
   Studio project. To try new data or settings, select **Use as base for new training** on a
   finished classifier. The preview shows what transfers. The original model stays intact;
   **Started from** on the new model card identifies the parent.

## Configure your installation

| Option | Setup |
|---|---|
| **DINO features** | Mount a compatible frozen encoder read-only at the same container path in the API and preparation worker. Set `SPARROW_FT_EMBEDDER=onnx:/weights/encoder.onnx` in `.env.service`, using your mounted path. The [weights mount example](docs/user-manual.md#3-model-weights) shows how to start Compose with an override. |
| **NVIDIA GPU** | Install the NVIDIA Container Toolkit, set `SPARROW_FT_GPU_PIPELINES=yolo_detector,conventional_classifier` in `.env.service`, then start the GPU profile below. Without a GPU worker, leave this setting empty. |
| **Sparrow Studio** | Use a Studio version with the Fine-Tune connector. Configure the Studio URL and shared service token on the Fine-Tune side, and the browser-facing Fine-Tune URL in Studio. See [both sides of the connection](docs/user-manual.md#5-sparrow-studio-integration-optional). |

```bash
# Add the CUDA worker to the existing CPU installation
docker compose -f compose.service.yml --env-file .env.service --profile gpu up -d --build
```

For shared access, set `SPARROW_FT_AUTH_DISABLED=0` and, behind HTTPS,
`SPARROW_FT_COOKIE_SECURE=1`. Apply the settings and create an account. In Bash:

```bash
docker compose -f compose.service.yml --env-file .env.service up -d
read -rs -p 'New admin password: ' password; echo
printf '%s\n' "$password" | docker compose -f compose.service.yml --env-file .env.service exec -T api sparrow-finetune create-user admin
unset password
```

## Development

The FastAPI service and workers share PostgreSQL and a data volume. For the API schema,
open [interactive API docs](http://127.0.0.1:8000/docs) on your running installation.
To run the tests from source with `uv` and Docker:

```bash
uv sync --extra dev --extra export --extra e2e
uv run --no-sync playwright install chromium
uv run --no-sync python -m pytest -q tests
```

See the [user manual](docs/user-manual.md) for operations, API routes and the separate legacy
YOLO workbench. For project help, see [Support](SUPPORT.md); report security issues through
the [security policy](SECURITY.md).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

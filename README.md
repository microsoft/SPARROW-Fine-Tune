# SPARROW Fine-Tune

A self-hosted service for fine-tuning wildlife camera-trap models. It imports labelled
images, helps label them with existing detectors and classifiers, trains a model, compares
it against the stock model, and exports it as an ONNX bundle. It can run on its own or
next to Sparrow Studio, which opens it from its
"Model fine-tune" page, sends it reviewed project data and installs the trained models.

| Pipeline | Task | Where it trains |
|---|---|---|
| `dinov3_linear` — DINOv3 linear probe | classification on detector crops | CPU (needs an embedder) |
| `conventional_classifier` — timm fine-tune (ResNet-50 by default) | classification on crops or whole images | GPU recommended |
| `yolo_detector` — YOLO11 object detector | detection | GPU recommended |

The service is a FastAPI app, a PostgreSQL database, and one or more workers that claim
jobs from the database. All of it runs from `compose.service.yml`.

## Quick start (Docker)

Requirements: Docker with Compose v2. For the GPU worker, an NVIDIA GPU with the NVIDIA
Container Toolkit.

```bash
cp .env.service.example .env.service          # set POSTGRES_PASSWORD at least
docker compose -f compose.service.yml --env-file .env.service up -d --build
curl -fsS http://127.0.0.1:8000/v1/health      # {"status": "ok", "database": "ok"}
```

Open <http://127.0.0.1:8000/> in a browser.

Add the GPU worker (trains `yolo_detector` and `conventional_classifier`):

```bash
# in .env.service: SPARROW_FT_GPU_PIPELINES=yolo_detector,conventional_classifier
docker compose -f compose.service.yml --env-file .env.service --profile gpu up -d
```

CPU-only machine: build with `TORCH_VARIANT=cpu` (smaller image, CPU torch):

```bash
TORCH_VARIANT=cpu docker compose -f compose.service.yml --env-file .env.service up -d --build
```

Require sign-in (sign-in is off by default because Sparrow Studio opens the app):

```bash
# in .env.service: SPARROW_FT_AUTH_DISABLED=0
docker compose -f compose.service.yml --env-file .env.service up -d
docker compose -f compose.service.yml exec api sparrow-finetune create-user admin
```

The API listens on `127.0.0.1:8000` by default. For access from other machines, put a TLS
reverse proxy in front and set `SPARROW_FT_COOKIE_SECURE=1`; do not publish the port
without sign-in.

Configuration, model weights, the Studio connection, the API and operations are covered in
the [user manual](docs/user-manual.md).

## Development

```bash
uv sync --extra dev --extra export
uv run --extra dev pytest -q tests          # needs Docker: the tests start a throwaway Postgres
uv run sparrow-finetune --help
```

The repository also contains the earlier single-user YOLO training workbench
(`src/sparrow_yolo`, `src/sparrow_web`, `Dockerfile`, `docker-compose.yml`). The service does
not depend on it; see the user manual's appendix.

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

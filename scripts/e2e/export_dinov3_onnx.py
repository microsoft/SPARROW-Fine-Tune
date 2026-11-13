"""Export a Hugging Face DINOv3 backbone to the ONNX image encoder ``OnnxEmbedder`` expects.

Input ``images`` float32 NCHW 224x224, ImageNet-normalised; output ``embedding`` (N, dim), the
pooled CLS feature. Used by the full Studio e2e test for ``SPARROW_FT_EMBEDDER=onnx:<path>``.

    uv run --with transformers python scripts/e2e/export_dinov3_onnx.py --out dinov3.onnx \
        [--model facebook/dinov3-vits16-pretrain-lvd1689m]

The DINOv3 weights are gated on Hugging Face; they must already be in the local cache or
reachable with a token that has access.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


class Pooled(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images):
        return self.backbone(pixel_values=images).pooler_output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoModel

    model = Pooled(AutoModel.from_pretrained(args.model).eval())
    x = torch.randn(2, 3, 224, 224)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (x,),
        str(args.out),
        input_names=["images"],
        output_names=["embedding"],
        # The TorchScript exporter mis-traces DINOv3's RoPE attention (max error ~0.1);
        # the dynamo exporter matches PyTorch to ~1e-6.
        dynamic_shapes={"images": {0: torch.export.Dim("n", min=1, max=64)}},
        dynamo=True,
    )
    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    (got,) = sess.run(None, {"images": x.numpy()})
    with torch.no_grad():
        want = model(x).numpy()
    delta = float(np.max(np.abs(got - want)))
    rel = delta / float(np.max(np.abs(want)))
    print(
        f"{args.out}: output {got.shape}, max |onnx - torch| = {delta:.2e} ({rel:.2e} relative)"
    )
    if rel > 1e-3:
        raise SystemExit("ONNX export does not match the PyTorch model")


if __name__ == "__main__":
    main()

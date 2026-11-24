"""Describe initialization from the producing job, not the stock comparison model."""


def starting_weights(pipeline, params, metadata, *, random_head=False):
    parent = metadata.get("parent_model_id") or params.get("parent_model_id")
    base = metadata.get("base_model_id") or params.get("base_model_id")
    if parent:
        return f"Finished model #{parent}"
    if base:
        return f"Imported base model #{base}"
    if pipeline == "dinov3_linear":
        encoder = (metadata.get("embedder") or {}).get("id")
        if encoder:
            head = "randomly initialized linear head" if random_head else "new linear head (initialization not recorded)"
            return f"Frozen encoder {encoder} · {head}"
    if pipeline == "conventional_classifier":
        arch = metadata.get("model_arch") or params.get("model")
        if arch and params.get("pretrained") is False:
            return f"{arch} · random initialization"
        if arch and params.get("pretrained") is True:
            source = "ImageNet" if arch in {"resnet50", "efficientnet_b0"} else "timm"
            return f"{arch} · {source} pretrained weights"
    if pipeline == "yolo_detector":
        model = metadata.get("base_model") or params.get("model")
        if model:
            return f"{model} · random initialization" if model.endswith((".yaml", ".yml")) else model
    return "Not recorded"

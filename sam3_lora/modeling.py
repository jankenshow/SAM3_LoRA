"""The only boundary that imports the vendored SAM3 model or downloads weights."""

import sys
from pathlib import Path

from .config import PROJECT_ROOT


def ensure_sam3_path():
    repository = PROJECT_ROOT / "vendor" / "sam3"
    if (repository / "sam3" / "__init__.py").is_file():
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))
    elif repository.exists() or (PROJECT_ROOT / ".gitmodules").is_file():
        raise FileNotFoundError(
            "SAM3 submodule is not initialized. Run: git submodule update --init --recursive"
        )


def find_bpe_path(package_file):
    """Resolve the vocabulary in an official checkout or an installed wheel."""
    package = Path(package_file).parent
    filename = "bpe_simple_vocab_16e6.txt.gz"
    for candidate in (package / "assets" / filename, package.parent / "assets" / filename):
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "SAM3 BPE vocabulary is missing; initialize the submodule or set model.bpe_path"
    )


def build_model(config, device="cpu", *, inference=False):
    model_config = config.get("model", {})
    checkpoint = model_config.get("checkpoint_path")
    allow_download = model_config.get("allow_download", False)
    if not checkpoint and not allow_download:
        raise ValueError(
            "Set model.checkpoint_path to local sam3.pt or explicitly enable model.allow_download"
        )
    if checkpoint and not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    ensure_sam3_path()
    import sam3
    from sam3.model_builder import build_sam3_image_model

    return build_sam3_image_model(
        device=str(device),
        compile=False,
        checkpoint_path=checkpoint,
        load_from_HF=allow_download,
        eval_mode=inference,
        bpe_path=model_config.get("bpe_path") or find_bpe_path(sam3.__file__),
    )


def build_adapted_model(config, device="cpu", *, inference=False, weights=None):
    from .adapters import LoRAConfig, apply_lora_to_model, load_lora_weights

    model = build_model(config, device, inference=inference)
    apply_lora_to_model(model, LoRAConfig.from_mapping(config["lora"], inference=inference))
    if weights:
        load_lora_weights(model, weights)
    model.to(device)
    return model.eval() if inference else model.train()

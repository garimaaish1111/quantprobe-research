"""Precision-aware model loading.

One function loads the *same* model at FP16, INT8, or INT4 so that the only
thing differing between the three arms of the study is the weight
representation. Everything else - tokenizer, seed, device, dtype of the
returned activations - is held fixed by construction.

Read this before changing anything here: INT8/INT4 quantize the WEIGHTS. The compute dtype, and therefore every hidden state this module
produces, is FP16.

`bitsandbytes` is imported lazily. It is CUDA-only in practice, and importing
it on a CUDA-less machine either fails or emits noise that would break every
unrelated script in this repo.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("quantprobe.models.loader")

#: The three arms of the study. Order matters only for reporting.
PRECISIONS: tuple[str, ...] = ("fp16", "int8", "int4")

#: Arms that require CUDA + bitsandbytes.
QUANTIZED_PRECISIONS: frozenset[str] = frozenset({"int8", "int4"})

#: Storage dtypes that prove bitsandbytes actually quantized the weights.
#: LLM.int8() writes torch.int8; NF4 packs two 4-bit values into torch.uint8.
QUANTIZED_STORAGE_DTYPES: frozenset[str] = frozenset({"int8", "uint8"})


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded for a reason the user must fix."""


def _torch():
    import torch

    return torch


def resolve_torch_dtype(name: str):
    """Map a config string like 'float16' to a torch dtype object."""
    torch = _torch()
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    key = str(name).lower().replace("torch.", "")
    if key not in table:
        raise ModelLoadError(
            f"unknown dtype {name!r}; expected one of {sorted(set(table))}"
        )
    return table[key]


def _dtype_kwarg_name() -> str:
    """Name of the load-dtype keyword for the installed transformers.

    transformers renamed `torch_dtype` to `dtype` during the 4.56 -> 5.x
    window. API surfaces are never guessed, so this is
    resolved from the installed version rather than assumed.
    """
    import transformers

    parts = transformers.__version__.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):  # pragma: no cover - odd dev versions
        return "dtype"
    if (major, minor) >= (4, 56):
        return "dtype"
    return "torch_dtype"


def cuda_available() -> bool:
    try:
        return _torch().cuda.is_available()
    except Exception:  # pragma: no cover
        return False


def resolve_device(requested: str | None = None) -> str:
    """Pick a device. 'auto' (the default) means CUDA if present, else CPU."""
    if requested and requested not in ("auto", None):
        return requested
    return "cuda" if cuda_available() else "cpu"


def hf_token() -> str | None:
    """Read an HF token from the environment. Never from a config file.

    A token in a YAML file gets committed. This reads HF_TOKEN or the
    huggingface_hub-standard HUGGING_FACE_HUB_TOKEN and nothing else.

    Whitespace is stripped and a blank result becomes None. A token pasted
    with a stray space or newline is otherwise truthy, sails past the
    "is a token present" guard, and fails 400 lines later as an opaque 401.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        raw = os.environ.get(var)
        if raw and raw.strip():
            return raw.strip()
    return None


def preflight_gated_access(hf_id: str, token: str | None) -> None:
    """Check the token can actually reach `hf_id` before loading anything.

    Having access and having a *token that carries* that access are different
    things, and the failure looks identical from the outside. HuggingFace's
    token UI defaults to "Fine-grained", and a fine-grained token does NOT
    reach gated repos unless the gated-repo permission is explicitly ticked.
    So an account showing ACCEPTED on the licence page still 401s.

    Without this, that surfaces as a ~100-line traceback ending in a generic
    "you are trying to access a gated repo. Make sure to have access" - which
    sends you back to the licence page you already completed, i.e. it points
    at the wrong thing.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:  # pragma: no cover
        return

    try:
        HfApi().model_info(hf_id, token=token)
    except Exception as exc:
        name = type(exc).__name__
        if "Gated" in name or "401" in str(exc) or "Unauthorized" in str(exc):
            raise ModelLoadError(
                f"HuggingFace rejected the token for {hf_id} (401).\n"
                f"\n"
                f"The token is set ({len(token or '')} chars), so this is not a "
                f"missing-token problem. Either:\n"
                f"\n"
                f"  1. The token is FINE-GRAINED and lacks gated-repo access.\n"
                f"     This is the usual cause. Your account can have the "
                f"licence ACCEPTED while the token still cannot read the repo.\n"
                f"     Fix: https://huggingface.co/settings/tokens -> create a "
                f"token of type 'Read' (not 'Fine-grained'), or edit the "
                f"fine-grained token and tick 'Read access to contents of all "
                f"public gated repos you can access'.\n"
                f"\n"
                f"  2. The token was mistyped, truncated, or revoked.\n"
                f"     Fix: create a fresh one and paste it again.\n"
                f"\n"
                f"  3. The licence really is still pending.\n"
                f"     Check: https://huggingface.co/settings/gated-repos\n"
                f"     It must say ACCEPTED, not PENDING.\n"
                f"\n"
                f"Underlying error: {name}: {str(exc)[:200]}"
            ) from exc
        # Anything else - offline, DNS, rate limit - is not ours to diagnose.
        logger.warning("could not preflight %s (%s); continuing", hf_id, name)


def build_quantization_config(precision: str, quant_cfg: dict) -> Any | None:
    """Build a `BitsAndBytesConfig` for INT8/INT4, or None for FP16.

    Args:
        precision: one of PRECISIONS.
        quant_cfg: the `quantization` block of the config, i.e. a mapping
            from precision name to that arm's settings.

    Returns:
        A BitsAndBytesConfig, or None when the arm needs no quantization.

    Raises:
        ModelLoadError: unknown precision, missing settings, or bitsandbytes
            not importable when it is required.
    """
    if precision not in PRECISIONS:
        raise ModelLoadError(
            f"unknown precision {precision!r}; expected one of {list(PRECISIONS)}"
        )

    if precision not in quant_cfg:
        raise ModelLoadError(
            f"config has no quantization settings for precision {precision!r}; "
            f"found settings for {sorted(quant_cfg)}"
        )

    settings = dict(quant_cfg[precision])

    if precision == "fp16":
        return None

    # --- Everything below needs CUDA + bitsandbytes. Fail with a message a
    #     human can act on, not an ImportError from three frames deep. ---
    if not cuda_available():
        raise ModelLoadError(
            f"precision {precision!r} needs bitsandbytes, which requires a CUDA "
            f"GPU. torch.cuda.is_available() is False on this machine. "
            f"Run the INT8/INT4 arms on Colab or Kaggle; the FP16 arm and all "
            f"of Stage B/C run fine on CPU."
        )

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise ModelLoadError(
            f"precision {precision!r} needs bitsandbytes but it is not "
            f"installed ({exc}). pip install bitsandbytes"
        ) from exc

    from transformers import BitsAndBytesConfig

    if precision == "int4":
        # This is the line that keeps activations FP16: the compute
        # dtype is FP16, so activations are FP16 even though weights are 4-bit.
        compute_dtype = settings.pop("bnb_4bit_compute_dtype", "float16")
        settings["bnb_4bit_compute_dtype"] = resolve_torch_dtype(compute_dtype)

    logger.info("quantization config for %s: %s", precision, settings)
    return BitsAndBytesConfig(**settings)


def load_model_and_tokenizer(
    cfg,
    precision: str,
    device: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load the model at one precision, plus its tokenizer and a metadata block.

    Args:
        cfg: a loaded Config. Must contain `model.hf_id` and `quantization`.
        precision: one of PRECISIONS.
        device: 'cuda', 'cpu', or None/'auto' to detect.

    Returns:
        (model, tokenizer, meta). `meta` is the provenance block describing
        exactly how this model was loaded, destined for a JSON sidecar.

    Raises:
        ModelLoadError: for any condition the user must fix - no CUDA for a
            quantized arm, gated model without a token, or an architecture
            that does not match the config's asserted layer count.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch = _torch()
    device = resolve_device(device)
    model_cfg = cfg["model"]
    hf_id = model_cfg["hf_id"]

    if model_cfg.get("gated", False):
        if hf_token() is None:
            raise ModelLoadError(
                f"{hf_id} is a gated repo and no token was found. Set HF_TOKEN in "
                f"the environment, and make sure the licence has been accepted at "
                f"https://huggingface.co/{hf_id}"
            )
        # A token being present is not the same as it working. Check now,
        # against the Hub, before we start downloading anything.
        preflight_gated_access(hf_id, hf_token())

    quant_config = build_quantization_config(precision, cfg["quantization"])

    load_kwargs: dict[str, Any] = {
        "quantization_config": quant_config,
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
    }

    dtype_kw = _dtype_kwarg_name()

    if quant_config is None:
        # --- FP16 arm ---
        if device == "cpu":
            # fp16 matmul on CPU is either unimplemented or pathologically
            # slow depending on the op. Silently producing garbage-slow
            # numbers would be worse than saying so.
            logger.warning(
                "FP16 requested on CPU. Loading in FLOAT32 instead: fp16 matmul "
                "on CPU is unsupported or extremely slow. This is fine for the "
                "smoke test but is NOT the fp16 arm of the study - that must "
                "run on CUDA."
            )
            load_kwargs[dtype_kw] = torch.float32
            effective_dtype = "float32 (cpu fallback)"
        else:
            load_kwargs[dtype_kw] = resolve_torch_dtype(
                cfg["quantization"]["fp16"].get("torch_dtype", "float16")
            )
            effective_dtype = str(load_kwargs[dtype_kw])
    else:
        # bitsandbytes places the model itself; passing a dtype here would
        # fight with it.
        effective_dtype = "fp16 compute (weights quantized)"
        # Hold the NON-quantized modules at the same dtype as the FP16 arm.
        #
        # Without this, transformers leaves them in the checkpoint's native
        # dtype - BF16 for Llama-3.2 and Qwen2.5. The FP16 arm casts
        # everything to FP16, so embeddings and layernorms would sit in BF16
        # on the INT arms and FP16 on the baseline. The arms would then differ
        # in TWO things - weight quantization AND the dtype of everything that
        # was never quantized - and the second is not what we are measuring.
        # Every arm must differ in exactly one thing.
        load_kwargs[dtype_kw] = resolve_torch_dtype(
            cfg["quantization"]["fp16"].get("torch_dtype", "float16")
        )

        # Pin the whole model to GPU 0.
        #
        # Without this, transformers may fall back to `device_map="auto"`
        # behaviour and offload some blocks to CPU when VRAM looks tight.
        # Offloaded blocks run in a different dtype on a different device,
        # so a layer's hidden state would then differ from the FP16 arm for
        # reasons that have nothing to do with quantization. That is exactly
        # the confound this whole study is trying to measure, so it must not
        # be allowed to happen silently. A 1B model at INT4 is ~1 GB; if it
        # does not fit on the GPU, we want the OOM, not the offload.
        load_kwargs["device_map"] = {"": 0}

    token = hf_token()
    if token:
        load_kwargs["token"] = token

    logger.info("loading %s at precision=%s device=%s", hf_id, precision, device)

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        **({"token": token} if token else {}),
    )

    # Decoder-only models frequently ship without a pad token. We need one to
    # batch. Reusing EOS is the standard fix; the attention mask means it is
    # never attended to, and Stage A finds the true last token from that mask
    # rather than trusting position -1.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("tokenizer had no pad_token; set pad_token = eos_token")

    model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)

    if quant_config is None:
        model = model.to(device)

    model.eval()

    meta = _describe_model(model, tokenizer, cfg, precision, device, effective_dtype)
    _assert_architecture(meta, model_cfg)
    _assert_single_device(model, precision)
    _assert_quantization_happened(meta, precision)
    return model, tokenizer, meta


def _assert_quantization_happened(meta: dict[str, Any], precision: str) -> None:
    """For INT8/INT4, verify the weights are ACTUALLY quantized.

    The single most dangerous silent failure in this project. If
    bitsandbytes no-ops - wrong version, a config key it does not recognise,
    a model whose modules it skips - `from_pretrained` still returns a
    perfectly working FP16 model. Every downstream number then comes out
    looking sane, the drift curve sits at zero, and the honest conclusion
    "quantization does not hurt the probe" would be entirely an artefact of
    never having quantized anything.

    The two schemes store weights differently, and BOTH count:

      - LLM.int8()  -> torch.int8   (one 8-bit value per byte)
      - NF4 4-bit   -> torch.uint8  (two 4-bit values packed per byte)

    Checking only for uint8 rejects a perfectly good INT8 arm. That is a fact
    about bitsandbytes' storage layout, not a heuristic - and it is why this
    check names the dtypes explicitly rather than substring-matching, since
    "uint8" contains "int8" and the two are easy to conflate.
    """
    if precision not in QUANTIZED_PRECISIONS:
        return

    dtype_counts: dict[str, int] = meta.get("param_dtype_counts", {})
    quantized_params = sum(
        count
        for dtype, count in dtype_counts.items()
        if dtype.rsplit(".", 1)[-1] in QUANTIZED_STORAGE_DTYPES
    )

    if quantized_params == 0:
        raise ModelLoadError(
            f"precision {precision!r} was requested but NO uint8 parameters exist "
            f"in the loaded model - parameter dtypes are {dtype_counts}. "
            f"bitsandbytes silently did not quantize anything, so this arm is "
            f"an FP16 model wearing an INT label. Every result derived from it "
            f"would be meaningless. Check the bitsandbytes version and that the "
            f"quantization_config reached from_pretrained."
        )

    logger.info(
        "%s quantization confirmed: %d quantized (uint8) parameter tensors, dtypes %s",
        precision, quantized_params, dtype_counts,
    )


def _assert_single_device(model, precision: str) -> None:
    """Refuse a model split across devices or offloaded to disk.

    See the note at `device_map={"": 0}` above. A partially-offloaded model
    produces hidden states that differ from the FP16 arm for placement
    reasons rather than quantization reasons, and nothing else in the
    pipeline would ever notice.
    """
    devices = {str(p.device) for p in model.parameters()}
    if len(devices) > 1:
        raise ModelLoadError(
            f"model at precision {precision!r} is split across devices {sorted(devices)}. "
            f"Mixed placement changes activations for reasons unrelated to "
            f"quantization and would confound the whole study. Free VRAM or use "
            f"a smaller model rather than allowing offload."
        )
    if any("meta" in d for d in devices):
        raise ModelLoadError(
            f"model at precision {precision!r} has parameters left on the meta "
            f"device - weights were never materialised. Devices: {sorted(devices)}"
        )


def _describe_model(
    model,
    tokenizer,
    cfg,
    precision: str,
    device: str,
    effective_dtype: str,
) -> dict[str, Any]:
    """Collect everything about this load that a sidecar should record."""
    config = model.config
    n_layers = getattr(config, "num_hidden_layers", None)
    d_model = getattr(config, "hidden_size", None)

    param_dtypes: dict[str, int] = {}
    for param in model.parameters():
        key = str(param.dtype)
        param_dtypes[key] = param_dtypes.get(key, 0) + 1

    return {
        "hf_id": cfg["model"]["hf_id"],
        "short_name": cfg["model"].get("short_name"),
        "precision": precision,
        "device": device,
        "effective_load_dtype": effective_dtype,
        "n_layers": n_layers,
        # +1 because output_hidden_states returns the embedding output first.
        "n_hidden_states": (n_layers + 1) if n_layers is not None else None,
        "d_model": d_model,
        "vocab_size": getattr(config, "vocab_size", None),
        "architecture": type(model).__name__,
        "param_dtype_counts": param_dtypes,
        "tokenizer_class": type(tokenizer).__name__,
        "pad_token": tokenizer.pad_token,
        "quantization_settings": (
            dict(cfg["quantization"][precision]) if precision in cfg["quantization"] else None
        ),
    }


def _assert_architecture(meta: dict[str, Any], model_cfg: dict) -> None:
    """Fail loudly if the loaded model is not the shape the config promised.

    If HF quietly serves a different revision with a different
    depth, every layer index in every figure silently means something else.
    """
    checks = [
        ("expected_n_layers", "n_layers"),
        ("expected_hidden_states", "n_hidden_states"),
        ("expected_d_model", "d_model"),
    ]
    problems = []
    for cfg_key, meta_key in checks:
        expected = model_cfg.get(cfg_key)
        if expected is None:
            continue
        actual = meta.get(meta_key)
        if actual != expected:
            problems.append(f"{meta_key}: config says {expected}, model reports {actual}")

    if problems:
        raise ModelLoadError(
            "loaded model does not match the architecture asserted in the "
            "config:\n  " + "\n  ".join(problems)
        )

"""Runtime compatibility patches for ``mbridge``.

transformers 5.x removed the flat ``rope_theta`` attribute from model
configs in favor of a ``rope_parameters`` dict (e.g. ``Qwen3MoeConfig``
under transformers 5.8.1 exposes ``rope_parameters={"rope_theta": ...,
"rope_type": ...}`` and raises ``AttributeError`` for ``rope_theta``).
mbridge 0.1.0 still reads ``self.hf_config.rope_theta`` when building
GPTModel args (``mbridge/core/llm_bridge.py``,
``LLMBridge._get_gptmodel_args``), which crashes model construction for
any bridge built from such a config — including ``Qwen3MoEBridge`` for
Qwen/Qwen3-30B-A3B.

``apply_mbridge_compat_patches()`` wraps ``LLMBridge._get_gptmodel_args``
so that a missing ``rope_theta`` is backfilled onto ``hf_config`` from
``rope_parameters`` before the original read. Backfilling the attribute
(rather than rewriting the return value) also repairs subclasses whose
overrides read ``self.hf_config.rope_theta`` via ``super()`` or after the
base method ran. The function is idempotent and a logged no-op when
mbridge is absent or already carries its own fallback.
"""

import functools
import inspect

from astraflow.train_worker.utils import logging

logger = logging.getLogger(__name__)

_PATCH_MARKER = "_astraflow_rope_theta_compat"


def _resolve_rope_theta(hf_config) -> float | None:
    """Return the rotary base from a transformers config, old or new style.

    Prefers the flat ``rope_theta`` attribute (transformers < 5), falling
    back to ``rope_parameters["rope_theta"]`` (transformers >= 5, where
    ``rope_parameters`` is a dict; tolerate attribute-style objects too).
    Returns None when neither form is present.
    """
    theta = getattr(hf_config, "rope_theta", None)
    if theta is not None:
        return theta
    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if rope_parameters is None:
        return None
    if isinstance(rope_parameters, dict):
        return rope_parameters.get("rope_theta")
    return getattr(rope_parameters, "rope_theta", None)


def apply_mbridge_compat_patches() -> None:
    """Patch mbridge for transformers 5.x configs. Idempotent, safe no-op.

    Must run before any ``mbridge.AutoBridge`` builds a model (the
    ``rope_theta`` read happens inside ``_get_gptmodel_args`` during
    ``get_model``). Importing this module's package from the Megatron
    engine before bridge construction satisfies that ordering.
    """
    try:
        from mbridge.core import llm_bridge
    except ImportError:
        logger.info("mbridge is not installed — skipping mbridge compat patches.")
        return

    original = llm_bridge.LLMBridge._get_gptmodel_args
    if getattr(original, _PATCH_MARKER, False):
        logger.info("mbridge rope_theta compat patch already applied — no-op.")
        return

    # Version gate: an mbridge that already falls back to rope_parameters
    # (upstream fix) needs no patch. Inspect the source rather than the
    # version string — the fix may be backported.
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError):
        source = ""
    if "rope_parameters" in source:
        logger.info(
            "mbridge already handles rope_parameters natively — "
            "skipping rope_theta compat patch."
        )
        return

    @functools.wraps(original)
    def _get_gptmodel_args_with_rope_fallback(self):
        if not hasattr(self.hf_config, "rope_theta"):
            theta = _resolve_rope_theta(self.hf_config)
            if theta is not None:
                # Backfill the legacy attribute so the original read (and
                # any other mbridge-side rope_theta read on this config)
                # succeeds.
                self.hf_config.rope_theta = theta
                logger.info(
                    "Backfilled hf_config.rope_theta=%s from "
                    "rope_parameters for %s (transformers 5.x compat).",
                    theta,
                    type(self.hf_config).__name__,
                )
        return original(self)

    setattr(_get_gptmodel_args_with_rope_fallback, _PATCH_MARKER, True)
    llm_bridge.LLMBridge._get_gptmodel_args = _get_gptmodel_args_with_rope_fallback
    logger.info(
        "Applied mbridge rope_theta compat patch "
        "(LLMBridge._get_gptmodel_args now falls back to rope_parameters)."
    )

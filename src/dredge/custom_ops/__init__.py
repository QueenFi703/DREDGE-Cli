import importlib
import logging

log = logging.getLogger(__name__)


def _fallback():
    def fused_wave(x, alpha=0.5):
        # simple reference Python implementation
        return alpha * x.sin() + (1 - alpha) * x

    return fused_wave


def _load():
    try:
        return importlib.import_module("dredge_custom_ops").fused_wave
    except (ImportError, ModuleNotFoundError) as e:
        log.info("Custom CUDA ops not available, using fallback: %s", e)
        return _fallback()


fused_wave = _load()

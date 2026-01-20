import os

try:
    from torch.utils.cpp_extension import load
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    load = None


def build_extensions():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required to build custom extensions. "
            "Please install PyTorch: pip install torch"
        )
    
    this_dir = os.path.dirname(__file__)
    csrc_dir = os.path.abspath(os.path.join(this_dir, "..", "..", "..", "csrc"))
    sources = [
        os.path.join(csrc_dir, "fused_wave.cpp"),
        os.path.join(csrc_dir, "fused_wave_kernel.cu"),
    ]
    return load(
        name="dredge_custom_ops",
        sources=sources,
        extra_cuda_cflags=["-lineinfo"],
        verbose=False,
    )


if __name__ == "__main__":
    build_extensions()

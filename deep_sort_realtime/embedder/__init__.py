def get_best_device(gpu=True):
    """Determine the best available torch device.

    Uses torch.accelerator (PyTorch >= 2.4) to detect any supported accelerator
    (CUDA, MPS, XPU, HPU, etc.). Falls back to manual checks for older PyTorch.
    Returns 'cpu' when gpu=False or no accelerator is available.

    Parameters
    ----------
    gpu : bool
        Whether to attempt to use an accelerator device. If False, always returns 'cpu'.

    Returns
    -------
    str
        Device type string, e.g. 'cuda', 'mps', 'xpu', or 'cpu'.
    """
    if not gpu:
        return "cpu"
    import torch
    # torch.accelerator (PyTorch >= 2.4) provides unified detection across all backends
    if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "is_available"):
        if torch.accelerator.is_available():
            try:
                return torch.accelerator.current_accelerator().type
            except RuntimeError:
                pass  # fall through to manual checks below
    # Fallback for PyTorch < 2.4
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

import torch


def get_best_device(gpu=True):
    """Determine the best available torch device.

    Priority: CUDA > MPS > CPU (when gpu=True).
    Returns 'cpu' when gpu=False.

    Parameters
    ----------
    gpu : bool
        Whether to attempt to use a GPU device. If False, always returns 'cpu'.

    Returns
    -------
    str
        One of 'cuda', 'mps', or 'cpu'.
    """
    if not gpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

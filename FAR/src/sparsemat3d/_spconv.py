"""Compatibility helpers for supported spconv API variants."""

from __future__ import annotations

from importlib import import_module

try:
    import spconv.pytorch as spconv
except ImportError:
    try:
        spconv = import_module("spconv")
    except ImportError as exc:
        raise ImportError(
            "sparsemat3d model execution requires spconv. "
            "Install a build compatible with the active PyTorch and CUDA stack."
        ) from exc


def replace_feature(sparse_tensor, updated_features):
    """Return a sparse tensor containing the updated feature matrix."""
    if hasattr(sparse_tensor, "replace_feature"):
        return sparse_tensor.replace_feature(updated_features)
    sparse_tensor.features = updated_features
    return sparse_tensor

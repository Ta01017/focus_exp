import numpy as np


def fuse_chroma(a, b):
    """Author's absolute-distance chroma fusion formula."""
    a = a.astype(np.float32); b = b.astype(np.float32)
    wa = np.abs(a - 128.0); wb = np.abs(b - 128.0)
    return (a * wa + b * wb) / (wa + wb + 1e-8)

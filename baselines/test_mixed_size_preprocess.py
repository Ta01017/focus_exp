import numpy as np
from PIL import Image

from metadata_dataset import synchronized_preprocess


def sample_from_base(base: np.ndarray) -> dict:
    def rgb(value):
        return Image.fromarray(np.repeat(value[:, :, None], 3, axis=2).astype(np.uint8), "RGB")
    return {
        "image_a": rgb(base), "image_b": rgb(base + 20), "target": rgb(base + 40),
        "a": rgb(base), "b": rgb(base + 20), "working_size": (base.shape[1], base.shape[0]),
    }


def channels(result):
    return tuple(np.asarray(result[key])[:, :, 0].astype(np.int16)
                 for key in ("image_a", "image_b", "target"))


def test_small_image_is_edge_padded_then_synchronously_cropped():
    base = np.arange(16, dtype=np.uint8).reshape(4, 4)
    result = synchronized_preprocess(
        sample_from_base(base), crop_size=128, mode="train", seed=7,
        hflip=True, vflip=True, rotate90=True,
    )
    a, b, gt = channels(result)
    assert a.shape == b.shape == gt.shape == (128, 128)
    assert np.array_equal(b - a, np.full_like(a, 20))
    assert np.array_equal(gt - a, np.full_like(a, 40))
    assert set(np.unique(a)).issubset(set(range(16)))


def test_large_non_square_image_is_cropped_without_whole_image_resize():
    base = np.tile(np.arange(100, dtype=np.uint8), (80, 1))
    result = synchronized_preprocess(
        sample_from_base(base), crop_size=(48, 32), mode="train", seed=11,
        hflip=False, vflip=False, rotate90=False,
    )
    a, b, gt = channels(result)
    assert a.shape == b.shape == gt.shape == (32, 48)
    assert np.array_equal(b - a, np.full_like(a, 20))
    assert np.array_equal(gt - a, np.full_like(a, 40))
    # A crop retains the original unit horizontal gradient; a 100->48 resize would not.
    assert np.all(np.diff(a, axis=1) == 1)


def test_diffusion_size_uses_same_padding_contract():
    base = np.arange(16, dtype=np.uint8).reshape(4, 4)
    result = synchronized_preprocess(
        sample_from_base(base), size=256, crop_size=256, mode="train", seed=3,
        hflip=True, vflip=True, rotate90=True, operation_order="crop_then_resize",
    )
    assert channels(result)[0].shape == (256, 256)

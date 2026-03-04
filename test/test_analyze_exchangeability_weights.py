import numpy as np

from scripts.analyze_exchangeability import _extract_weights_from_artifacts


def _f32_to_bf16_raw(arr: np.ndarray) -> np.ndarray:
    bits_u32 = arr.astype(np.float32).view(np.uint32)
    bits_u16 = (bits_u32 >> 16).astype(np.uint16)
    return bits_u16.view(np.dtype('V2'))


def _bf16_raw_to_f32(raw: np.ndarray) -> np.ndarray:
    bits_u16 = np.frombuffer(raw.tobytes(), dtype=np.uint16).reshape(raw.shape)
    bits_u32 = bits_u16.astype(np.uint32) << 16
    return bits_u32.view(np.float32)


def test_extract_weights_from_v2_bfloat16_artifact(tmp_path):
    group_dir = tmp_path / 'group_0'
    artifact_dir = group_dir / 'artifacts'
    artifact_dir.mkdir(parents=True)

    step = 123
    original = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 7.0
    raw = _f32_to_bf16_raw(original)
    np.savez(artifact_dir / f'first_layer_{step}.npz', first_layer_weights=raw)

    loaded = _extract_weights_from_artifacts(str(group_dir), step)
    expected = _bf16_raw_to_f32(raw)

    assert loaded.dtype == np.float32
    assert loaded.shape == original.shape
    assert np.allclose(loaded, expected)

import jax.numpy as jnp
import numpy as np

import scripts.analyze_exchangeability as analyze_exchangeability
from scripts.analyze_exchangeability import _activation_similarity_matrix
from scripts.analyze_exchangeability import _extract_weights_from_artifacts
from scripts.analyze_exchangeability import _save_similarity_distributions
from scripts.analyze_exchangeability import _similarity_npz_path
from src.experiment.exchangeability_utils import cosine_similarity_matrix


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


def test_save_similarity_distributions_writes_compressed_npz(tmp_path):
    similarity_dir = tmp_path / 'similarity_cache'
    width = 32
    step = 4096
    representation = 'weights'
    within_real = np.linspace(0.1, 0.9, 11, dtype=np.float64)
    across_real = np.linspace(0.0, 1.0, 17, dtype=np.float64)

    out_path = _save_similarity_distributions(
        similarity_output_dir=str(similarity_dir),
        dataset='imagenet',
        width=width,
        images_seen=step,
        representation=representation,
        within_real=within_real,
        across_real=across_real,
    )

    expected_path = _similarity_npz_path(str(similarity_dir), width, step, representation)
    assert out_path == expected_path

    loaded = np.load(out_path)
    assert loaded['within_real'].dtype == np.float32
    assert loaded['across_real'].dtype == np.float32
    assert np.allclose(loaded['within_real'], within_real.astype(np.float32))
    assert np.allclose(loaded['across_real'], across_real.astype(np.float32))
    assert int(loaded['width']) == width
    assert int(loaded['images_seen']) == step
    assert str(loaded['representation']) == representation


def test_activation_similarity_matrix_retries_smaller_chunks_after_oom(monkeypatch, capsys):
    call_sizes = []

    def fake_conv_init_features(kernel, batch_x):
        batch_size = int(batch_x.shape[0])
        call_sizes.append(batch_size)
        if batch_size > 1:
            raise ValueError('RESOURCE_EXHAUSTED: synthetic oom')

        base = np.asarray(batch_x, dtype=np.float32).reshape((batch_size, -1))[:, :1]
        offset = np.asarray(kernel, dtype=np.float32).reshape((-1, kernel.shape[-1])).sum(axis=0, keepdims=True)
        return jnp.asarray(base + offset, dtype=jnp.float32)

    monkeypatch.setattr(analyze_exchangeability, '_conv_init_features', fake_conv_init_features)

    member_variables = [
        {'params': {'conv_init': {'kernel': np.asarray([[[[0.0, 1.0]]]], dtype=np.float32)}}},
        {'params': {'conv_init': {'kernel': np.asarray([[[[1.0, 0.0]]]], dtype=np.float32)}}},
    ]
    probe_loader = [
        (
            np.asarray(
                [
                    [[ [1.0] ]],
                    [[ [2.0] ]],
                    [[ [3.0] ]],
                    [[ [4.0] ]],
                ],
                dtype=np.float32,
            ),
            None,
        )
    ]

    sim = _activation_similarity_matrix(
        member_variables=member_variables,
        width=2,
        probe_loader=probe_loader,
        activation_chunk_size=0,
        progress_label='test',
    )

    expected_features_0 = np.asarray(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ],
        dtype=np.float32,
    )
    expected_features_1 = np.asarray(
        [
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 3.0],
            [5.0, 4.0],
        ],
        dtype=np.float32,
    )
    expected = cosine_similarity_matrix(
        np.concatenate([expected_features_0.T, expected_features_1.T], axis=0),
        np.concatenate([expected_features_0.T, expected_features_1.T], axis=0),
    )

    assert 4 in call_sizes
    assert 2 in call_sizes
    assert call_sizes.count(1) >= 4
    assert np.allclose(sim, expected)

    captured = capsys.readouterr()
    assert 'activation OOM at chunk size 4; retrying with chunk size 2' in captured.out
    assert 'activation OOM at chunk size 2; retrying with chunk size 1' in captured.out


def test_activation_similarity_matrix_reuses_cached_chunk_size(monkeypatch, capsys):
    call_sizes = []

    def fake_conv_init_features(kernel, batch_x):
        batch_size = int(batch_x.shape[0])
        call_sizes.append(batch_size)
        if batch_size > 2:
            raise ValueError('RESOURCE_EXHAUSTED: synthetic oom')

        base = np.asarray(batch_x, dtype=np.float32).reshape((batch_size, -1))[:, :1]
        offset = np.asarray(kernel, dtype=np.float32).reshape((-1, kernel.shape[-1])).sum(axis=0, keepdims=True)
        return jnp.asarray(base + offset, dtype=jnp.float32)

    monkeypatch.setattr(analyze_exchangeability, '_conv_init_features', fake_conv_init_features)

    member_variables = [
        {'params': {'conv_init': {'kernel': np.asarray([[[[0.0, 1.0]]]], dtype=np.float32)}}},
        {'params': {'conv_init': {'kernel': np.asarray([[[[1.0, 0.0]]]], dtype=np.float32)}}},
    ]
    probe_loader = [
        (
            np.asarray(
                [
                    [[[1.0]]],
                    [[[2.0]]],
                    [[[3.0]]],
                    [[[4.0]]],
                ],
                dtype=np.float32,
            ),
            None,
        )
    ]
    chunk_cache = {}
    cache_key = ('imagenet', 2, 2)

    _activation_similarity_matrix(
        member_variables=member_variables,
        width=2,
        probe_loader=probe_loader,
        activation_chunk_size=0,
        progress_label='test',
        activation_chunk_size_cache=chunk_cache,
        activation_chunk_cache_key=cache_key,
    )

    assert 4 in call_sizes
    assert chunk_cache[cache_key] == 2

    call_sizes.clear()
    _activation_similarity_matrix(
        member_variables=member_variables,
        width=2,
        probe_loader=probe_loader,
        activation_chunk_size=0,
        progress_label='test',
        activation_chunk_size_cache=chunk_cache,
        activation_chunk_cache_key=cache_key,
    )

    assert 4 not in call_sizes
    assert call_sizes.count(2) >= 2

    captured = capsys.readouterr()
    assert 'test reusing cached activation chunk size 2' in captured.out


def test_resolve_shuffle_stats_workers_respects_slurm_cpu_allocation(monkeypatch):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '16')
    monkeypatch.setattr(analyze_exchangeability.os, 'cpu_count', lambda: 64)

    assert analyze_exchangeability._resolve_shuffle_stats_workers(0) == 15
    assert analyze_exchangeability._resolve_shuffle_stats_workers(7) == 7

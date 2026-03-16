import numpy as np
import pytest

from src.experiment.exchangeability_utils import (
    abs_cosine_similarity_matrix,
    build_member_ids,
    cosine_similarity_matrix,
    extract_across_values,
    extract_within_values,
    flatten_permute_reshape_indices,
    ks_w1_stats,
    ks_w1_stats_against_sorted_reference,
    make_target_points,
    shuffled_similarity_values,
    shuffled_similarity_values_batched,
    w1_distance,
    w1_distance_against_sorted_reference,
)



def test_make_target_points_includes_horizon():
    targets = make_target_points(1000, [100, 500, 1500])
    assert targets == [100, 500, 1000]



def test_cosine_similarity_preserves_sign():
    x = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=np.float64,
    )
    sim = cosine_similarity_matrix(x, x)
    assert np.allclose(np.diag(sim), 1.0)
    assert sim[0, 1] == pytest.approx(-1.0)


def test_abs_cosine_similarity_identity():
    x = np.eye(3, dtype=np.float64)
    sim = abs_cosine_similarity_matrix(x, x)
    assert np.allclose(np.diag(sim), 1.0)
    assert np.allclose(sim[0, 1], 0.0)



def test_pair_extraction_counts():
    num_members = 3
    width = 2
    total = num_members * width
    sim = np.arange(total * total, dtype=np.float64).reshape(total, total)
    sim = (sim + sim.T) / 2.0
    member_ids = build_member_ids(num_members, width)

    across = extract_across_values(sim, member_ids)
    within = extract_within_values(sim, member_ids)

    expected_across = (num_members * (num_members - 1) // 2) * (width * width)
    expected_within = num_members * (width * (width - 1) // 2)

    assert across.size == expected_across
    assert within.size == expected_within



def test_flatten_permute_shape():
    rng = np.random.default_rng(0)
    idx = flatten_permute_reshape_indices(4, 5, rng)
    assert idx.shape == (4, 5)
    assert sorted(idx.reshape(-1).tolist()) == list(range(20))



def test_ks_w1_stats_returns_metrics():
    x = np.random.default_rng(0).normal(size=256)
    y = np.random.default_rng(1).normal(loc=0.5, size=256)
    stats = ks_w1_stats(x, y)
    assert 'ks_distance' in stats
    assert 'ks_pvalue' in stats
    assert 'w1_distance' in stats
    assert stats['ks_distance'] >= 0.0


def test_batched_shuffle_matches_scalar_sequence():
    num_members = 4
    width = 3
    batch_size = 5
    total = num_members * width

    base_rng = np.random.default_rng(123)
    sim = base_rng.normal(size=(total, total))
    sim = (sim + sim.T) / 2.0

    scalar_rng = np.random.default_rng(321)
    scalar_across = []
    scalar_within = []
    for _ in range(batch_size):
        across, within = shuffled_similarity_values(sim, num_members, width, scalar_rng)
        scalar_across.append(across)
        scalar_within.append(within)

    batched_rng = np.random.default_rng(321)
    batched_across, batched_within = shuffled_similarity_values_batched(
        sim,
        num_members,
        width,
        batched_rng,
        batch_size=batch_size,
    )

    assert np.allclose(batched_across, np.stack(scalar_across))
    assert np.allclose(batched_within, np.stack(scalar_within))


def test_ks_w1_stats_matches_scipy_large_auto_asymp():
    scipy_stats = pytest.importorskip('scipy.stats')
    ks_2samp = scipy_stats.ks_2samp
    wasserstein_distance = scipy_stats.wasserstein_distance

    rng = np.random.default_rng(42)
    x = rng.normal(size=12000)
    y = rng.normal(loc=0.25, size=13000)

    got = ks_w1_stats(x, y)
    expected_ks = ks_2samp(x, y, alternative='two-sided', mode='auto')
    expected_w1 = wasserstein_distance(x, y)

    assert got['ks_distance'] == pytest.approx(float(expected_ks.statistic))
    assert got['ks_pvalue'] == pytest.approx(float(expected_ks.pvalue))
    assert got['w1_distance'] == pytest.approx(float(expected_w1))


def test_w1_distance_matches_scipy_with_repeated_values():
    scipy_stats = pytest.importorskip('scipy.stats')
    wasserstein_distance = scipy_stats.wasserstein_distance

    x = np.array([0.0, 0.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float64)
    y = np.array([-1.0, 0.0, 0.0, 0.5, 2.0, 3.0], dtype=np.float64)

    got = w1_distance(x, y)
    expected = float(wasserstein_distance(x, y))

    assert got == pytest.approx(expected)


def test_w1_distance_sorted_reference_matches_full():
    rng = np.random.default_rng(9)
    reference = rng.normal(size=4000)
    sample = rng.normal(loc=0.2, size=3500)

    assert w1_distance_against_sorted_reference(np.sort(reference), sample) == pytest.approx(
        w1_distance(reference, sample)
    )


def test_sorted_reference_path_matches_full_stats():
    rng = np.random.default_rng(7)
    reference = rng.normal(size=15000)
    sample = rng.normal(loc=-0.1, size=9000)

    full = ks_w1_stats(reference, sample)
    cached = ks_w1_stats_against_sorted_reference(np.sort(reference), sample)

    assert cached['ks_distance'] == pytest.approx(full['ks_distance'])
    assert cached['ks_pvalue'] == pytest.approx(full['ks_pvalue'])
    assert cached['w1_distance'] == pytest.approx(full['w1_distance'])

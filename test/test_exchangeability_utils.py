import numpy as np

from src.experiment.exchangeability_utils import (
    abs_cosine_similarity_matrix,
    build_member_ids,
    extract_across_values,
    extract_within_values,
    flatten_permute_reshape_indices,
    ks_w1_stats,
    make_target_points,
    shuffled_similarity_values,
    shuffled_similarity_values_batched,
)



def test_make_target_points_includes_horizon():
    targets = make_target_points(1000, [100, 500, 1500])
    assert targets == [100, 500, 1000]



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

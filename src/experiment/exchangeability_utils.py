from __future__ import annotations

import numpy as np

try:
    from scipy.stats import distributions, ks_2samp, norm
except Exception:  # pragma: no cover
    distributions = None
    ks_2samp = None
    norm = None


AUTO_EXACT_MAX_N = 10000



def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return x / safe_norms


def make_target_points(target_images_seen: int, p_targets_images_seen: list[int]) -> list[int]:
    values = sorted({int(v) for v in p_targets_images_seen if int(v) > 0})
    if not values:
        values = [target_images_seen]
    values = [v for v in values if v <= target_images_seen]
    if not values or values[-1] != target_images_seen:
        values.append(int(target_images_seen))
    return values



def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = normalize_rows(a)
    b_n = normalize_rows(b)
    return a_n @ b_n.T


def abs_cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(cosine_similarity_matrix(a, b))



def build_member_ids(num_members: int, width: int) -> np.ndarray:
    return np.repeat(np.arange(num_members), width)



def extract_across_values(similarity_matrix: np.ndarray, member_ids: np.ndarray) -> np.ndarray:
    idx = np.arange(similarity_matrix.shape[0])
    ii, jj = np.triu_indices(similarity_matrix.shape[0], k=1)
    across_mask = member_ids[ii] != member_ids[jj]
    return similarity_matrix[ii[across_mask], jj[across_mask]]



def extract_within_values(similarity_matrix: np.ndarray, member_ids: np.ndarray) -> np.ndarray:
    ii, jj = np.triu_indices(similarity_matrix.shape[0], k=1)
    within_mask = member_ids[ii] == member_ids[jj]
    return similarity_matrix[ii[within_mask], jj[within_mask]]



def flatten_permute_reshape_indices(num_members: int, width: int, rng: np.random.Generator) -> np.ndarray:
    total = num_members * width
    perm = rng.permutation(total)
    return perm.reshape((num_members, width))



def shuffled_similarity_values(
    similarity_matrix: np.ndarray,
    num_members: int,
    width: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    across_batch, within_batch = shuffled_similarity_values_batched(
        similarity_matrix=similarity_matrix,
        num_members=num_members,
        width=width,
        rng=rng,
        batch_size=1,
    )
    return across_batch[0], within_batch[0]


def shuffled_similarity_values_batched(
    similarity_matrix: np.ndarray,
    num_members: int,
    width: int,
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if batch_size <= 0:
        raise ValueError('batch_size must be positive.')

    total = num_members * width
    flat_permutations = np.empty((batch_size, total), dtype=np.int64)
    for batch_idx in range(batch_size):
        flat_permutations[batch_idx] = rng.permutation(total)

    member_groups = flat_permutations.reshape((batch_size, num_members, width))

    tri_i, tri_j = np.triu_indices(width, k=1)
    within = similarity_matrix[member_groups[:, :, tri_i], member_groups[:, :, tri_j]]
    within = within.reshape((batch_size, -1))

    member_i, member_j = np.triu_indices(num_members, k=1)
    group_i = member_groups[:, member_i, :]
    group_j = member_groups[:, member_j, :]
    across = similarity_matrix[group_i[:, :, :, None], group_j[:, :, None, :]]
    across = across.reshape((batch_size, -1))

    return across, within



def _ks_distance_numpy(x: np.ndarray, y: np.ndarray) -> float:
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    all_values = np.concatenate([x_sorted, y_sorted])
    cdf_x = np.searchsorted(x_sorted, all_values, side='right') / x_sorted.size
    cdf_y = np.searchsorted(y_sorted, all_values, side='right') / y_sorted.size
    return float(np.max(np.abs(cdf_x - cdf_y)))



def _w1_distance_from_sorted(x_sorted: np.ndarray, y_sorted: np.ndarray) -> float:
    if x_sorted.size == 0 or y_sorted.size == 0:
        raise ValueError('Both samples must be non-empty.')

    x_size = int(x_sorted.size)
    y_size = int(y_sorted.size)
    x_idx = 0
    y_idx = 0
    cdf_x = 0.0
    cdf_y = 0.0
    prev_value = None
    w1_distance = 0.0

    while x_idx < x_size or y_idx < y_size:
        next_x = x_sorted[x_idx] if x_idx < x_size else np.inf
        next_y = y_sorted[y_idx] if y_idx < y_size else np.inf
        current_value = float(next_x if next_x <= next_y else next_y)

        if prev_value is not None:
            delta = current_value - prev_value
            if delta != 0.0:
                w1_distance += abs(cdf_x - cdf_y) * delta

        while x_idx < x_size and float(x_sorted[x_idx]) <= current_value:
            x_idx += 1
        while y_idx < y_size and float(y_sorted[y_idx]) <= current_value:
            y_idx += 1

        cdf_x = x_idx / x_size
        cdf_y = y_idx / y_size
        prev_value = current_value

    return float(w1_distance)


def w1_distance(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        raise ValueError('Both samples must be non-empty.')
    return _w1_distance_from_sorted(np.sort(x), np.sort(y))


def w1_distance_against_sorted_reference(reference_sorted: np.ndarray, sample: np.ndarray) -> float:
    if reference_sorted.size == 0 or sample.size == 0:
        raise ValueError('Both samples must be non-empty.')
    return _w1_distance_from_sorted(reference_sorted, np.sort(sample))


def _ks_w1_from_sorted(x_sorted: np.ndarray, y_sorted: np.ndarray) -> tuple[float, float]:
    if x_sorted.size == 0 or y_sorted.size == 0:
        raise ValueError('Both samples must be non-empty.')

    x_size = int(x_sorted.size)
    y_size = int(y_sorted.size)
    x_idx = 0
    y_idx = 0
    cdf_x = 0.0
    cdf_y = 0.0
    prev_value = None
    ks_distance = 0.0
    w1_distance = 0.0

    while x_idx < x_size or y_idx < y_size:
        next_x = x_sorted[x_idx] if x_idx < x_size else np.inf
        next_y = y_sorted[y_idx] if y_idx < y_size else np.inf
        current_value = float(next_x if next_x <= next_y else next_y)

        if prev_value is not None:
            delta = current_value - prev_value
            if delta != 0.0:
                w1_distance += abs(cdf_x - cdf_y) * delta

        while x_idx < x_size and float(x_sorted[x_idx]) <= current_value:
            x_idx += 1
        while y_idx < y_size and float(y_sorted[y_idx]) <= current_value:
            y_idx += 1

        cdf_x = x_idx / x_size
        cdf_y = y_idx / y_size
        ks_distance = max(ks_distance, abs(cdf_x - cdf_y))
        prev_value = current_value

    return float(ks_distance), float(w1_distance)


def _ks_pvalue_from_distance(ks_distance: float, n1: int, n2: int) -> float:
    if ks_2samp is None:
        return float('nan')
    if max(n1, n2) <= AUTO_EXACT_MAX_N:
        return float('nan')
    if distributions is None:
        return float('nan')

    m = float(max(n1, n2))
    n = float(min(n1, n2))
    effective_n = m * n / (m + n)
    pvalue = float(distributions.kstwo.sf(ks_distance, np.round(effective_n)))
    return float(np.clip(pvalue, 0.0, 1.0))


def ks_w1_stats_from_sorted(x_sorted: np.ndarray, y_sorted: np.ndarray) -> dict[str, float]:
    if x_sorted.size == 0 or y_sorted.size == 0:
        raise ValueError('Both samples must be non-empty.')

    ks_distance, w1_distance = _ks_w1_from_sorted(x_sorted, y_sorted)

    if ks_2samp is not None and max(x_sorted.size, y_sorted.size) <= AUTO_EXACT_MAX_N:
        ks_res = ks_2samp(x_sorted, y_sorted, alternative='two-sided', mode='auto')
        ks_pvalue = float(ks_res.pvalue)
    else:
        ks_pvalue = _ks_pvalue_from_distance(ks_distance, int(x_sorted.size), int(y_sorted.size))

    return {
        'ks_distance': ks_distance,
        'ks_pvalue': ks_pvalue,
        'w1_distance': w1_distance,
    }


def ks_w1_stats_against_sorted_reference(reference_sorted: np.ndarray, sample: np.ndarray) -> dict[str, float]:
    if reference_sorted.size == 0 or sample.size == 0:
        raise ValueError('Both samples must be non-empty.')
    sample_sorted = np.sort(sample)
    return ks_w1_stats_from_sorted(reference_sorted, sample_sorted)



def ks_w1_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if x.size == 0 or y.size == 0:
        raise ValueError('Both samples must be non-empty.')

    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    return ks_w1_stats_from_sorted(x_sorted, y_sorted)



def two_sided_sigma_from_p(p_value: float) -> float:
    if not np.isfinite(p_value) or p_value <= 0.0:
        p_value = 1e-300
    if p_value >= 1.0:
        return 0.0
    if norm is None:  # pragma: no cover
        return float('nan')
    return float(norm.isf(p_value / 2.0))

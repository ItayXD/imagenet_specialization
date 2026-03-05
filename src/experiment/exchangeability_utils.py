from __future__ import annotations

import numpy as np

try:
    from scipy.stats import ks_2samp, norm, wasserstein_distance
except Exception:  # pragma: no cover
    ks_2samp = None
    norm = None
    wasserstein_distance = None



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



def abs_cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = normalize_rows(a)
    b_n = normalize_rows(b)
    return np.abs(a_n @ b_n.T)



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



def _w1_numpy(x: np.ndarray, y: np.ndarray) -> float:
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    q = np.linspace(0.0, 1.0, min(x_sorted.size, y_sorted.size), endpoint=False)
    xq = np.quantile(x_sorted, q, method='linear')
    yq = np.quantile(y_sorted, q, method='linear')
    return float(np.mean(np.abs(xq - yq)))



def ks_w1_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if x.size == 0 or y.size == 0:
        raise ValueError('Both samples must be non-empty.')

    if ks_2samp is not None:
        ks_res = ks_2samp(x, y, alternative='two-sided', mode='auto')
        ks_distance = float(ks_res.statistic)
        ks_pvalue = float(ks_res.pvalue)
    else:  # pragma: no cover
        ks_distance = _ks_distance_numpy(x, y)
        ks_pvalue = float('nan')

    if wasserstein_distance is not None:
        w1 = float(wasserstein_distance(x, y))
    else:  # pragma: no cover
        w1 = _w1_numpy(x, y)

    return {
        'ks_distance': ks_distance,
        'ks_pvalue': ks_pvalue,
        'w1_distance': w1,
    }



def two_sided_sigma_from_p(p_value: float) -> float:
    if not np.isfinite(p_value) or p_value <= 0.0:
        p_value = 1e-300
    if p_value >= 1.0:
        return 0.0
    if norm is None:  # pragma: no cover
        return float('nan')
    return float(norm.isf(p_value / 2.0))

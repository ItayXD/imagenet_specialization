from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    input_shape: tuple[int, int, int]
    num_classes: int
    stem_type: str


_DATASET_SPECS = {
    'imagenet': DatasetSpec(
        dataset='imagenet',
        input_shape=(224, 224, 3),
        num_classes=1_000,
        stem_type='imagenet',
    ),
    'cifar5m': DatasetSpec(
        dataset='cifar5m',
        input_shape=(32, 32, 3),
        num_classes=10,
        stem_type='cifar',
    ),
}


def get_dataset_spec(dataset: str) -> DatasetSpec:
    try:
        return _DATASET_SPECS[str(dataset)]
    except KeyError as exc:
        supported = ', '.join(sorted(_DATASET_SPECS))
        raise ValueError(f'Unsupported dataset "{dataset}". Expected one of: {supported}.') from exc

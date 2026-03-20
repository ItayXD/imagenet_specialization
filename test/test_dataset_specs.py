import pytest

from src.experiment.dataset_specs import get_dataset_spec


def test_imagenet_dataset_spec():
    spec = get_dataset_spec('imagenet')
    assert spec.input_shape == (224, 224, 3)
    assert spec.num_classes == 1000
    assert spec.stem_type == 'imagenet'


def test_cifar5m_dataset_spec():
    spec = get_dataset_spec('cifar5m')
    assert spec.input_shape == (32, 32, 3)
    assert spec.num_classes == 10
    assert spec.stem_type == 'cifar'


def test_unknown_dataset_spec_errors():
    with pytest.raises(ValueError):
        get_dataset_spec('unknown')

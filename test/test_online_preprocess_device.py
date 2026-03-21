from src.run.OnlinePreprocessDevice import OnlinePreprocessDevice


class _DummyPreprocessDevice(OnlinePreprocessDevice):
    def load_data(self, data_params):
        del data_params
        raise NotImplementedError


def test_prep_vd_uses_single_process_loader(monkeypatch, tmp_path):
    observed = {}

    class _Loader:
        def __iter__(self):
            return iter([('images', 'labels')])

    def _fake_make_dataloader(dataset, *args, **kwargs):
        observed['dataset'] = dataset
        observed['args'] = args
        observed['kwargs'] = kwargs
        return _Loader()

    monkeypatch.setattr('src.run.OnlinePreprocessDevice.make_dataloader', _fake_make_dataloader)

    preprocess = _DummyPreprocessDevice(str(tmp_path), {'root_dir': 'data'})
    preprocess._prep_vd([1, 2, 3])

    assert observed['dataset'] == [1, 2, 3]
    assert observed['args'] == ()
    assert observed['kwargs']['batch_size'] == 3
    assert observed['kwargs']['num_workers'] == 0
    assert preprocess.val_data == ('images', 'labels')

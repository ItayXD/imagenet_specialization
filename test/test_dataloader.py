from src.run.dataloader import make_dataloader


def test_make_dataloader_uses_spawn_when_workers_enabled(monkeypatch):
    observed = {}

    def _fake_dataloader(dataset, *args, **kwargs):
        observed['dataset'] = dataset
        observed['args'] = args
        observed['kwargs'] = kwargs
        return object()

    monkeypatch.setattr('src.run.dataloader.DataLoader', _fake_dataloader)

    make_dataloader([1, 2, 3], 4, num_workers=2, shuffle=True, drop_last=False)

    assert observed['dataset'] == [1, 2, 3]
    assert observed['args'] == (4,)
    assert observed['kwargs']['num_workers'] == 2
    assert observed['kwargs']['multiprocessing_context'] == 'spawn'


def test_make_dataloader_keeps_single_process_loader_simple(monkeypatch):
    observed = {}

    def _fake_dataloader(dataset, *args, **kwargs):
        observed['dataset'] = dataset
        observed['args'] = args
        observed['kwargs'] = kwargs
        return object()

    monkeypatch.setattr('src.run.dataloader.DataLoader', _fake_dataloader)

    make_dataloader([1, 2, 3], 4, num_workers=0, shuffle=False, drop_last=False)

    assert observed['dataset'] == [1, 2, 3]
    assert observed['args'] == (4,)
    assert observed['kwargs']['num_workers'] == 0
    assert 'multiprocessing_context' not in observed['kwargs']

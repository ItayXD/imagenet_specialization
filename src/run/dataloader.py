from torch.utils.data import DataLoader


def make_dataloader(dataset, *args, num_workers: int = 0, **kwargs):
    if num_workers > 0:
        kwargs.setdefault('multiprocessing_context', 'spawn')
    return DataLoader(dataset, *args, num_workers=num_workers, **kwargs)

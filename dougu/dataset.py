from pathlib import Path

from torch.utils.data import DataLoader

from . import (
    Configurable,
    WithLog,
    torch_cached_property,
    cached_property,
    SubclassRegistry,
    )


def get_sampler(split, distributed=False, rank=0):
    if distributed:
        import torch.utils.data.distributed as dist
        return dist.DistributedSampler(split, rank=rank)
    else:
        return None


class TensorDictDataset():
    def __init__(self, tensor_dict):
        self.tensors = tensor_dict

    def __getitem__(self, index):
        return {k: tensor[index] for k, tensor in self.tensors.items()}

    def __len__(self):
        return len(next(iter(self.tensors.values())))

    def split(self, split_size):
        import torch
        split_idxs = torch.arange(len(self)).split(split_size)
        return tuple(self[idxs] for idxs in split_idxs)


class Dataset(SubclassRegistry, Configurable, WithLog):
    args = [
        ('--data-dir', dict(type=Path, default='data')),
        ('--cache-dir', dict(type=Path, default='cache')),
        ('--max-train-inst', dict(type=int)),
        ('--max-dev-inst', dict(type=int)),
        ('--max-test-inst', dict(type=int)),
        ]

    def has_train_data(self):
        return True

    @property
    def conf_fields(self):
        fields = []
        for split_name in ['train', 'dev', 'test']:
            if self.get_max_inst(split_name) is not None:
                fields.append(self.max_inst_field(split_name))
        return fields

    @property
    def dir_name(self):
        return self.__class__.__name__.lower()

    def split_fname(self, split_name):
        raise NotImplementedError()

    def split_file(self, split_name):
        fname = self.split_fname(split_name)
        return self.conf.data_dir / self.dir_name / fname

    @cached_property
    def raw(self):
        return self.load_raw_data()

    def load_raw_data(self):
        return {
            split_name:
                self.load_raw_split(split_name)[:self.get_max_inst(split_name)]
            for split_name in self.split_names}

    def max_inst_field(self, split_name):
        return f'max_{split_name}_inst'

    def get_max_inst(self, split_name):
        return getattr(self.conf, self.max_inst_field(split_name), None)

    def load_raw_split(self, split_name):
        return self.load_raw_file(self.split_file(split_name))

    def load_raw_file(self, split_file):
        raise NotImplementedError()

    @torch_cached_property
    def tensors(self):
        return self.tensorize()

    @property
    def collate_fn(self):
        from torch.utils.data.dataloader import default_collate
        return default_collate

    def tensorize(self):
        return {
            split_name: self.tensorize_split(split)
            for split_name, split in self.raw.items()}

    def tensorize_split(self, split):
        raise NotImplementedError()

    def log_size(self):
        split_names = getattr(self, 'loader_names', self.split_names)
        for split_name in split_names:
            self.log_split_size(split_name)

    def log_split_size(self, split_name):
        split = getattr(self, split_name, None)
        if split is not None:
            msg = f'{len(split)} {split_name} instances'
        else:
            msg = ''
        loader_name = split_name + '_loader'
        if loader_name is not None:
            loader = getattr(self, loader_name)
            if loader is not None:
                msg += f' | {len(loader)} batches'
        if msg:
            self.log(msg)

    def metrics(self, prefix):
        return {}

    @property
    def batch_size(self):
        return getattr(self.conf, 'batch_size', 32)

    @property
    def eval_batch_size(self):
        return getattr(self.conf, 'eval_batch_size', 64)


class TrainDevTest():
    @property
    def split_names(self):
        return ['train', 'dev', 'test']

    def split(self, split_name):
        return TensorDictDataset(self.tensors[split_name])

    @cached_property
    def train(self):
        return self.split('train')

    @cached_property
    def dev(self):
        return self.split('dev')

    @cached_property
    def test(self):
        return self.split('test')

    @cached_property
    def train_loader(self):
        sampler = get_sampler(
            self.train,
            distributed=getattr(self.conf, 'distributed', False),
            rank=getattr(self.conf, 'local_rank', 0),
            )
        return DataLoader(
            self.train,
            self.batch_size,
            sampler=sampler,
            collate_fn=self.collate_fn)

    @property
    def dev_loader(self):
        return self.eval_loader(self.dev)

    @property
    def test_loader(self):
        return self.eval_loader(self.test)

    def eval_loader(self, tensorized_data):
        return DataLoader(
            tensorized_data,
            self.eval_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn)


class EvalOnTrain():
    @property
    def split_names(self):
        return ['train']

    @property
    def loader_names(self):
        return ['train', 'dev']

    @cached_property
    def train(self):
        return TensorDictDataset(self.tensors['train'])

    @cached_property
    def train_loader(self):
        sampler = get_sampler(
            self.train,
            distributed=getattr(self.conf, 'distributed', False),
            rank=getattr(self.conf, 'local_rank', 0),
            )
        return DataLoader(
            self.train,
            self.batch_size,
            sampler=sampler,
            shuffle=True if sampler is None else None,
            collate_fn=self.collate_fn)

    @cached_property
    def train_loader_no_shuffle(self):
        return DataLoader(
            self.train,
            self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn)

    @cached_property
    def train_eval(self):
        import torch
        from torch.utils.data import Subset
        idxs = torch.randperm(len(self.train))[:self.conf.max_test_inst]
        return Subset(self.train, idxs)

    @property
    def dev_loader(self):
        return DataLoader(
            self.train_eval,
            self.eval_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn)


TrainOnly = EvalOnTrain

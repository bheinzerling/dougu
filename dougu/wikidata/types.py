from pathlib import Path

import torch

from dougu import (
    flatten,
    cached_property,
    file_cached_property,
    )

from .wikidata_attribute import WikidataAttribute


class WikidataTypes(WikidataAttribute):
    key = 'P31'
    args = [
        ('--wikidata-types-fname',
            dict(type=Path, default='P31.object.most_freq_1000')),
        ]

    def log_size(self):
        tensor = self.tensor.float()
        self.log(f'{tensor.shape}')
        self.log(f'{tensor.sum(dim=1).mean().item():.2f} types per instance')
        self.log(f'{tensor.sum(dim=1).max().item():.2f} types max')

    @property
    def allowed_types_file(self):
        return self.wikidata.data_dir / self.conf.wikidata_types_fname

    @cached_property
    def allowed_types(self):
        return set(self.type_enc.labels)

    @cached_property
    def n_types(self):
        return len(self.allowed_types)

    @file_cached_property
    def type_enc(self):
        from dougu.codecs import LabelEncoder
        return LabelEncoder.from_file(self.allowed_types_file, to_torch=True)

    @cached_property
    def raw(self):
        def get_types(inst):
            types = set(inst.get(self.key, []))
            return list(filter(self.allowed_types.__contains__, types))

        id2types = {
            inst['id']: get_types(inst) for inst in self.wikidata.raw['train']}
        assert set(id2types.keys()) == set(self.entity_ids)
        return [id2types[entity_id] for entity_id in self.entity_ids]

    @file_cached_property
    def tensor(self):
        typess = self.raw
        n_types = torch.tensor(list(map(len, typess)))
        types = list(flatten(typess))
        types_enc = self.type_enc.transform(types)

        entity_idxs = torch.arange(len(typess))
        row_idxs = entity_idxs.repeat_interleave(n_types)
        col_idxs = types_enc
        idxs = torch.stack([row_idxs, col_idxs])
        vals = torch.ones_like(col_idxs, dtype=torch.int8)
        size = torch.Size((len(typess), self.n_types))
        return torch.sparse.ByteTensor(idxs, vals, size).to_dense()

    def tensorize(self):
        return {
            split_name: self.tensorize_split(split)
            for split_name, split in self.raw.items()}

    @property
    def color_tensor_source(self):
        return self.tensor

    @property
    def color_tensor(self):
        from sklearn.decomposition import PCA
        c = PCA(n_components=3).fit_transform(self.color_tensor_source)
        c = (c - c.min()) / c.ptp()
        color = (c * 255).astype(int)
        return color

from pathlib import Path

import torch

from dougu import (
    flatten,
    dict_load,
    groupby,
    cached_property,
    file_cached_property,
    )

from .wikidata_attribute import WikidataAttribute


class WikidataTypes(WikidataAttribute):
    key = 'P31'
    args = [
        ('--wikidata-types-fname',
            dict(type=Path, default='P31.object.mincount_100')),
        ('--wikidata-types-counts-fname',
            dict(type=Path, default='P31.object.counts')),
        ('--wikidata-types-label-fname',
            dict(type=Path, default='P31.object.labels_en')),
        ]

    @property
    def conf_fields(self):
        return super().conf_fields + [
            'wikidata_types_fname',
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

    def of(self, inst, *args, **kwargs):
        types = set(inst.get(self.key, []))
        return list(filter(self.allowed_types.__contains__, types))

    @cached_property
    def counts(self):
        f = self.wikidata.data_dir / self.conf.wikidata_types_counts_fname
        d = dict_load(f)
        return {k: v for k, v in d.items() if k in self.allowed_types}

    @cached_property
    def raw(self):
        id2types = {
            inst['id']: self.of(inst) for inst in self.wikidata.raw['train']}
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

    @cached_property
    def label2type_ids(self):
        keys, values = zip(*self.type_id2label.items())
        return groupby(values, keys)

    @cached_property
    def type_id2label(self):
        f = self.wikidata.data_dir / self.conf.wikidata_types_label_fname
        d = dict_load(f, splitter='\t')
        return {k: v for k, v in d.items() if k in self.allowed_types}

    def search(self, query, counts=False):
        import re
        pattern = re.compile(query)

        def maybe_add_counts(type_ids):
            if counts:
                return [
                    (type_id, self.counts[type_id])
                    for type_id in type_ids
                    ]
            return type_ids

        matches = [
            (label, maybe_add_counts(type_ids))
            for label, type_ids in self.label2type_ids.items()
            if re.search(pattern, label)
            ]
        return matches

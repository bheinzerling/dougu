from pathlib import Path

import torch

from tqdm import tqdm

from dougu import (
    flatten,
    cached_property,
    file_cached_property,
    avg,
    lines,
    )

from .wikidata_attribute import WikidataAttribute


class WikidataRelations(WikidataAttribute):
    key = 'relation'
    no_rel = 'no_rel'

    args = WikidataAttribute.args + [
        ('--relations-object-counts-fname',
            dict(type=Path, default='relation.obj.counts')),
        ('--relations-object-min-count', dict(type=int, default=50)),
        ]

    def log_size(self):
        n_rels = list(map(len, self.raw))
        self.log(
            f'{sum(n_rels)} rels | '
            f'min: {min(n_rels)} | avg: {avg(n_rels)} | max: {max(n_rels)}')

    @cached_property
    def preds(self):
        return [self.no_rel] + list(self.wikidata.property_id2label.keys())

    @cached_property
    def n_preds(self):
        return len(self.preds)

    @file_cached_property(fname_tpl='pred_enc.{conf_str}.pkl')
    def pred_enc(self):
        from dougu.codecs import LabelEncoder
        return LabelEncoder(to_torch=True, device='cpu').fit(self.preds)

    @cached_property
    def frequent_object_ids(self):
        object_ids = []
        f = self.wikidata.data_dir / self.conf.relations_object_counts_fname
        for line in lines(f):
            object_id, count = line.split('\t')
            count = int(count)
            if count >= self.conf.relations_object_min_count:
                object_ids.append(object_id)
            else:
                break
        return object_ids

    @cached_property
    def entity_ids(self):
        return set(self.wikidata.entity_ids)

    @cached_property
    def allowed_object_ids(self):
        # return self.entity_ids | set(self.frequent_object_ids)
        return self.entity_ids

    def of(self, inst, *args, **kwargs):
        head = inst['id']
        dummy_rel = {self.no_rel: [head]}
        return [
            [head, pred, tail]
            for pred, tails in inst.get(self.key, dummy_rel).items()
            for tail in tails
            if tail in self.allowed_object_ids]

    @cached_property
    def raw(self):
        id2relations = {
            inst['id']: self.of(inst)
            for inst in tqdm(self.wikidata.raw_iter)}
        assert set(id2relations.keys()) == set(self.entity_ids)
        assert len(set(self.entity_ids) - set(id2relations.keys())) == 0
        return [id2relations[entity_id] for entity_id in self.entity_ids]

    @file_cached_property
    def tensor(self):
        relations = flatten(self.raw)
        return self.encode(relations)

    def encode(self, relations):
        heads, preds, tails = zip(*relations)
        tensors = [
            self.wikidata.entity_id_enc.transform(heads),
            self.pred_enc.transform(preds),
            self.wikidata.entity_id_enc.transform(tails)]
        return torch.stack(tensors, dim=1)

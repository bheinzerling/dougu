from pathlib import Path

from dougu import (
    jsonlines_load,
    file_cached_property,
    dict_load,
    )
from dougu.dataset import Dataset, TrainOnly


class Wikidata(Dataset, TrainOnly):
    args = [
        ('--wikidata-dir-name', dict(type=Path, default='wikidata')),
        ('--wikidata-top-n', dict(type=int, default=10000)),
        ('--wikidata-fname-tpl',
            dict(type=str, default='instances.kilt_top{top}.jsonl')),
        ('--wikidata-label-lang', dict(type=str, default='en')),
        ('--wikidata-property-label-file', dict(
            type=Path,
            default='data/wikidata/property_label.en')),
        ]

    @property
    def conf_fields(self):
        fields = [
            'wikidata_top_n',
            'wikidata_label_lang',
            ]
        return fields

    def split_file(self, split_name):
        fname = self.conf.wikidata_fname_tpl.format(
            top=self.conf.wikidata_top_n)
        return self.conf.data_dir / self.conf.wikidata_dir_name / fname

    def load_raw_split(self, split_name):
        split_file = self.split_file(split_name)
        return list(jsonlines_load(split_file))

    @file_cached_property
    def entity_ids(self):
        return [inst['id'] for inst in self.raw['train']]

    @file_cached_property
    def entity_id_enc(self):
        from dougu.codecs import LabelEncoder
        return LabelEncoder(
            backend='dict', to_torch=True, device='cpu').fit(self.entity_ids)

    @file_cached_property
    def entity_id2label(self):
        label_lang = self.conf.wikidata_label_lang
        return {
            inst['id']: inst['label'].get(label_lang, inst['id'])
            for inst in self.raw['train']
            }

    def entity_idxs2labels(self, entity_idxs):
        entity_ids = self.entity_id_enc.inverse_transform(entity_idxs)
        if isinstance(entity_ids, str):
            entity_id = entity_ids
            return self.entity_id2label.get(entity_id, entity_id)
        return [
            self.entity_id2label.get(entity_id, entity_id)
            for entity_id in entity_ids]

    @file_cached_property
    def property_id2label(self):
        return dict_load(self.conf.wikidata_property_label_file, splitter='\t')

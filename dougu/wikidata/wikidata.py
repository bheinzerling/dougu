from pathlib import Path

from dougu import (
    jsonlines_load,
    cached_property,
    file_cached_property,
    dict_load,
    )
from dougu.dataset import Dataset, TrainOnly

from .wikidata_attribute import WikidataAttribute
from .numeric_attributes import WikidataNumericAttributes
from .types import WikidataTypes
from .label import WikidataLabel, WikidataAliases
from .relations import WikidataRelations
from .description import WikidataDescription
from .popularity import WikidataPopularity
from .subclass_of import SubclassOf
from .db import DB


class Wikidata(Dataset, TrainOnly):
    args = [
        ('--wikidata-dir-name', dict(type=Path, default='wikidata')),
        ('--wikidata-top-n', dict(type=int)),
        ('--wikidata-fname', dict(type=str)),
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
            'wikidata_fname_tpl',
            'wikidata_label_lang',
            'wikidata_fname',
            ]
        return fields

    @property
    def data_dir(self):
        return self.conf.data_dir / self.conf.wikidata_dir_name

    def split_file(self, split_name):
        if self.conf.wikidata_fname:
            fname = self.conf.wikidata_fname
        else:
            fname = self.conf.wikidata_fname_tpl.format(
                top=self.conf.wikidata_top_n)
        return self.conf.data_dir / self.conf.wikidata_dir_name / fname

    def load_raw_split(self, split_name):
        split_file = self.split_file(split_name)
        return list(jsonlines_load(split_file))

    @property
    def raw_iter(self):
        split_file = self.split_file('train')
        yield from jsonlines_load(split_file)

    @file_cached_property
    def entity_ids(self):
        return [inst['id'] for inst in self.raw['train']]

    @cached_property
    def enttiy_id2entity(self):
        return {inst['id']: inst for inst in self.raw['train']}

    def __getitem__(self, entity_id):
        return self.enttiy_id2entity[entity_id]

    @cached_property
    def n_entities(self):
        return len(self.entity_ids)

    @file_cached_property
    def entity_id_enc(self):
        from dougu.codecs import LabelEncoder
        return LabelEncoder(
            backend='dict', to_torch=True, device='cpu').fit(self.entity_ids)

    @file_cached_property
    def entity_id2label(self):
        label_lang = self.conf.wikidata_label_lang
        if self.raw_data_loaded:
            instances = self.raw['train']
        else:
            instances = self.raw_iter
        return {
            inst['id']: inst.get('label', {}).get(label_lang, inst['id'])
            for inst in instances
            }

    def entity_ids2labels(self, entity_ids):
        return [self.entity_id2label.get(id, id) for id in entity_ids]

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

    @cached_property
    def numericattributes(self):
        return WikidataNumericAttributes(self.conf, self)

    @cached_property
    def types(self):
        return WikidataTypes(self.conf, self)

    @cached_property
    def label(self):
        return WikidataLabel(self.conf, self)

    @cached_property
    def aliases(self):
        return WikidataAliases(self.conf, self)

    @cached_property
    def relations(self):
        return WikidataRelations(self.conf, self)

    @cached_property
    def description(self):
        return WikidataDescription(self.conf, self)

    @cached_property
    def popularity(self):
        return WikidataPopularity(self.conf, self)

    @cached_property
    def attribute_classes(self):
        return WikidataAttribute.get_subclasses()

    def attributes(self, inst):
        label_lang = self.conf.wikidata_label_lang

        def to_dict(attr, key):
            if isinstance(attr, dict):
                return attr
            return {key: attr}

        def get_attr(inst, attr_cls):
            key = attr_cls.__name__.lstrip('Wikidata').lower()
            attr = getattr(self, key).of(inst, label_lang)
            return to_dict(attr, key)

        attrs = dict(
            item
            for attr_cls in self.attribute_classes
            for item in get_attr(inst, attr_cls).items()
            )
        attrs['id'] = inst['id']
        return attrs

    @cached_property
    def subclass_of(self):
        return SubclassOf(self.conf)

    @cached_property
    def db(self):
        return DB(self.conf, self)

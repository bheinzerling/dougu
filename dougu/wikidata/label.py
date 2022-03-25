from dougu import (
    cached_property,
    )
from dougu.transformer_mixin import WithTransformerEncoder

from .wikidata_attribute import WikidataAttribute


class WikidataLabel(WikidataAttribute, WithTransformerEncoder):
    key = 'label'

    @property
    def conf_fields(self):
        fields = super().conf_fields
        fields.extend([
            'max_seq_len',
            'transformer',
            ])
        return fields

    @cached_property
    def raw(self):
        lang = self.wikidata.conf.wikidata_label_lang
        return [self.of(inst, lang) for inst in self.wikidata.raw['train']]

    def of(self, inst, lang):
        return inst.get(self.key, {}).get(lang, inst['id'])

    @cached_property
    def tensor(self):
        return self.encode_texts(self.raw)


class WikidataAliases(WikidataLabel):
    key = 'alias'

    args = WikidataLabel.args + [
        ('--wikidaata-max-aliases', dict(type=int, default=32)),
        ]

    def of(self, inst, lang):
        aliases = inst.get(self.key, {})
        label = [self.wikidata.label.of(inst, lang)]
        return aliases.get(lang, label)

    @cached_property
    def tensor(self):
        raise NotImplementedError('todo')

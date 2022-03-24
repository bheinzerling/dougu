from dougu import (
    Configurable,
    WithLog,
    )


class WikidataAttribute(Configurable, WithLog):
    def __init__(self, conf, wikidata, *args, **kwargs):
        super().__init__(conf, *args, **kwargs)
        self.wikidata = wikidata

    @property
    def conf_fields(self):
        fields = [
            'wikidata_top_n',
            ]
        return fields

    @property
    def entity_ids(self):
        return self.wikidata.entity_ids

    @property
    def n_entities(self):
        return self.wikidata.n_entities

    @property
    def raw(self):
        raise NotImplementedError()

    @property
    def tensor(self):
        raise NotImplementedError()

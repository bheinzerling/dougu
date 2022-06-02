from dougu import (
    Configurable,
    cached_property,
    file_cached_property,
    WithLog,
    flatten,
    )


class DB(Configurable, WithLog):
    def __init__(self, conf, wikidata, *args, **kwargs):
        super().__init__(conf, *args, **kwargs)
        self.wikidata = wikidata
        self.make_tables()

    @property
    def name(self):
        return 'wikidata'

    @property
    def conf_fields(self):
        return self.wikidata.conf_fields

    @cached_property
    def engine(self):
        from sqlalchemy import create_engine
        return create_engine(f'sqlite:///cache/{self.conf_str}.sqlite')

    def make_tables(self):
        try:
            if self.name in self.engine.table_names():
                raise ValueError()
            with self.engine.connect() as con:
                self.df.to_sql(self.name, con)
                self.df_relations.to_sql(self.name + '_relations', con)
            mode = 'created'
        except ValueError:
            mode = 'found existing'
        self.log(f'{mode} DB {self.engine.url}')

    @file_cached_property
    def df(self):
        import pandas as pd
        from tqdm import tqdm

        def to_simple_values(attributes):
            sep = ';'
            types = attributes['types']
            attributes['types'] = sep.join(types) + sep

            attributes['aliases'] = sep.join(
                a for a in attributes['aliases'] if sep not in a)

            del attributes['relations']

            return attributes

        attrss = map(self.wikidata.attributes, tqdm(self.wikidata.raw_iter))
        data = list(map(to_simple_values, attrss))
        df = pd.DataFrame(data)

        return df

    @file_cached_property
    def df_relations(self):
        import pandas as pd
        rels = flatten(self.wikidata.relations.raw)
        return pd.DataFrame(rels, columns=['subj', 'pred', 'obj'])

    def query(self, sql):
        with self.engine.connect() as con:
            rows = con.execute(sql)
            for row in rows:
                yield dict(row)

    def __call__(self, sql):
        return list(self.query(sql))

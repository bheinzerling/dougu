import shutil
import pandas as pd
from dougu import now_str


class Results():
    def __init__(self, file, *, index, result_fields, backup=True):
        self.file = file
        self.key = "df"
        columns = index + result_fields
        self.columns = columns
        self.default_min_itemsize = 100
        self.min_itemsize = {c: self.default_min_itemsize for c in columns}
        self.index = index
        self.backup = backup

    def __enter__(self):
        if self.backup:
            file_bak = str(self.file) + "." + now_str() + ".bak"
            try:
                shutil.copy(self.file, file_bak)
            except FileNotFoundError:
                pass
        self.store = pd.HDFStore(self.file)
        if self.key not in self.store or self.store[self.key] is None:
            self.dataframe = pd.DataFrame(columns=self.columns)
        return self

    def __exit__(self, *args):
        self.store.close()

    def append(self, row, index=True):
        row = pd.DataFrame([row])
        self.store.append(
            self.key, row, format="table", data_columns=True,
            index=index,
            min_itemsize=self.min_itemsize)

    @property
    def dataframe(self):
        self.store[self.key]
        df = self.store[self.key][self.columns]
        df.set_index(self.index, inplace=True)
        df.sort_index(level=df.index.names)
        return df

    @dataframe.setter
    def dataframe(self, df):
        self.store.put(
            self.key, df, format="table", data_columns=True,
            min_itemsize=self.min_itemsize)

    @property
    def df(self):
        return self.dataframe.reset_index()

    def add_column(
            self, column_name,
            *,
            default_value=None, values=None, new_column_order=None,
            min_itemsize=None):
        df = self.df
        if default_value is not None:
            assert values is None
            values = [default_value] * len(df)
        assert values is not None
        df[column_name] = values
        min_itemsize = min_itemsize or self.default_min_itemsize
        self.min_itemsize[column_name] = min_itemsize
        if new_column_order:
            df = df[new_column_order]
        else:
            new_column_order = self.columns + [column_name]
        self.columns = new_column_order
        self.dataframe = df

    def n_done(self, conf_key):
        import warnings
        warnings.simplefilter(
            action='ignore', category=pd.errors.PerformanceWarning)
        try:
            return len(self.dataframe.loc[conf_key])
        except (KeyError, TypeError):
            return 0

    def mean_std_table(
            self, col_name, outfile, group_index=None, group_sort_key=None):
        grouped = self.dataframe.groupby(level=group_index or self.index)
        r = grouped.mean()
        r["std"] = grouped.std()
        r["trials"] = grouped.size()
        sort_key = col_name + "__$sort"
        r[sort_key] = r[col_name]
        r[col_name] = r[[col_name, "std"]].applymap(
            lambda v: "{:.1f}".format(100 * v).rjust(4)
            ).apply(' ± '.join, 1)
        if group_sort_key:
            s = r.sort_values(
                sort_key, ascending=False).groupby(level=group_sort_key)
            with outfile.with_suffix(".sorted").open("w") as out:
                for k, v in s.groups.items():
                    g = s.get_group(k)
                    del g["std"]
                    del g[sort_key]
                    out.write(g.to_string() + "\n\n")
        del r["std"]
        del r[sort_key]
        with outfile.with_suffix(".tex").open("w") as out:
            out.write(r.to_latex())
        with outfile.with_suffix(".txt").open("w") as out:
            out.write(r.to_string())

from pathlib import Path
import csv


class WALS(dict):
    wals_file = Path(__file__).parent / Path("data/wals/language.csv")

    def __init__(self, wals_file=None, key="iso_code"):
        if wals_file:
            self.wals_file = wals_file
        self.key = key
        self._load()

    def _load(self):
        with self.wals_file.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row[self.key]:
                    self[row[self.key]] = row

    def feature_counts(self, langs=None):
        from collections import Counter
        return Counter([
            k
            for lang, feats in self.items()
            for k, v in feats.items()
            if (langs is None or lang in langs) and (v and k[0].isdigit())])

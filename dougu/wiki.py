from pathlib import Path

from .io import dict_load


wiki_iso_file = (
    Path(__file__).parent / Path("data/wikipedia/wiki_iso_wikiname_langname"))


wiki2iso = dict_load(wiki_iso_file)
lang2wiki = dict_load(wiki_iso_file, key_index=2, value_index=0, splitter="\t")
wiki2lang = dict_load(wiki_iso_file, key_index=0, value_index=2, splitter="\t")

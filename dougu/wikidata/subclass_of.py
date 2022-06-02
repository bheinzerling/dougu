from pathlib import Path
from collections import Counter, defaultdict
from functools import cache

import networkx as nx

from dougu import (
    Configurable,
    cached_property,
    json_load,
    dict_load,
    flatten,
    lines,
    )
from dougu.graph import graph_from_edges


class SubclassOf(Configurable):
    args = [
        ('--wikidata-dir', dict(
            type=Path, default='data/wikidata')),
        ('--closure-file', dict(
            type=Path, default='entity_attr/subclass_of.closures.json')),
        ('--edge-file-dir', dict(
            type=Path, default='entity_attr/subclass_of')),
        ('--label-file', dict(
            type=Path, default='subclass_of.object.label_en')),
        ]

    @cached_property
    def node2children(self):
        node2children = defaultdict(set)
        for source, target in self.edges:
            if source == target:
                continue
            node2children[target].add(source)
        return node2children

    @cached_property
    def node2parents(self):
        node2parents = defaultdict(set)
        for source, target in self.edges:
            if source == target:
                continue
            node2parents[source].add(target)
        return node2parents

    @cached_property
    def node2parent_closure(self):
        return json_load(self.conf.wikidata_dir / self.conf.closure_file)

    @cached_property
    def node2count(self):
        return Counter(flatten(self.node2parent_closure.values()))

    @cached_property
    def edges(self):
        def load_edges(edge_file_dir):
            for line in flatten(map(lines, edge_file_dir.iterdir())):
                edge = line.split('\t')
                if len(edge) == 3:
                    yield edge[0], edge[2]

        edges_file = self.conf.wikidata_dir / self.conf.edge_file_dir
        return list(load_edges(edges_file))

    @cached_property
    def id2label(self):
        f = self.conf.wikidata_dir / self.conf.label_file
        return dict_load(f, splitter='\t')

    def labels(self, ids):
        if isinstance(ids, str):
            return self.id2label.get(ids, ids)
        return [self.id2label.get(id, id) for id in ids]

    def sorted_by_count(self, ids):
        return sorted(ids, key=self.node2count.__getitem__, reverse=True)

    @property
    def root(self):
        return 'Q35120'  # entity

    @cached_property
    def node2depth(self):
        depth = 1
        node2depth = defaultdict(list)
        node2depth[self.root] = depth
        added = {self.root}
        children = self.node2children[self.root]
        while children:
            depth += 1
            print(depth, len(children))
            grandchildren = set()
            for child in children:
                node2depth[child] = depth
                added.add(child)
                grandchildren.update(self.node2children[child])
            children = grandchildren - added
        return node2depth

    @cached_property
    def grggph(self):
        return graph_from_edges(self.edges)

    @cache
    def depth(self, node):
        # + 1 because root node has depth 1
        return nx.shortest_path_length(
            self.graph, source=node, target=self.root) + 1

    @cache
    def path_to_root(self, node):
        return nx.shortest_path(
            self.graph, source=node, target=self.root)

    @cache
    def least_common_subsumer(self, node1, node2):
        ancestors1 = set(self.node2parent_closure[node1])
        ancestors2 = set(self.node2parent_closure[node2])
        common_ancestors = list(ancestors1 & ancestors2)
        depths = list(map(self.depth, common_ancestors))
        # choose more frequent node as lowest common ancestor in case of ties
        counts = list(map(g.node2count.__getitem__, common_ancestors))
        depth, _, node = sorted(zip(
            depths, counts, common_ancestors), reverse=True)[0]
        # the subclass graph is a rooted directed graph and not a tree, so
        # there can be least common subsumers that are further away from
        # the root than the two nodes to be subsumed. In this case
        # choose the root as least common subsumer
        max_node_depth = max(self.depth(node1), self.depth(node2))
        if depth > max_node_depth:
            depth = 1
            node = self.root
        return node, depth

    @cache
    def wu_palmer_sim(self, node1, node2):
        d1 = self.depth(node1)
        d2 = self.depth(node2)
        lcs, lcs_depth = self.least_common_subsumer(node1, node2)
        sim = 2 * lcs_depth / (d1 + d2)
        return dict(
            sim=sim,
            lcs=lcs,
            lcs_depth=lcs_depth,
            lcs_label=self.labels(lcs),
            depth1=d1,
            node1_label=self.labels(node1),
            depth2=d2,
            node2_label=self.labels(node2),
            )


if __name__ == "__main__":
    conf = Configurable.get_conf()
    g = SubclassOf(conf)
    breakpoint()

def graph_from_edges(edges, directed=True):
    import networkx as nx
    # if len(edges[0]) == 3:
    #     edges = [
    #         (source, target, {'label': label})
    #         for source, label, target in edges
    #         ]

    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    g.add_edges_from(edges)
    return g

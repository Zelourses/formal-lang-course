import pyformlang
from pyformlang.cfg import Terminal
from scipy.sparse import lil_matrix

import networkx as nx
from typing import *

import copy

from project.onfh_task06 import cfg_to_weak_normal_form


def cfpq_with_matrix(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> set[tuple[int, int]]:

    start_nodes = graph.nodes if start_nodes is None else start_nodes
    final_nodes = graph.nodes if final_nodes is None else final_nodes
    cfg = cfg_to_weak_normal_form(cfg)
    n = len(graph.nodes)

    epsilons = set()
    terminals = {}
    mults = {}

    # changing the sparsity structure of a csr_matrix is expensive. lil_matrix is more efficient (c) scipy warning
    productions: dict[Any, lil_matrix] = {}

    for p in cfg.productions:
        pLen = len(p.body)
        if pLen == 0:
            epsilons.add(p.head.value)
        elif pLen == 1 and isinstance(p.body[0], Terminal):
            terminals.setdefault(p.body[0].value, set()).add(p.head.value)
        elif pLen == 2:
            mults.setdefault(p.head.value, set()).add(
                (p.body[0].value, p.body[1].value)
            )

        productions[p.head.value] = lil_matrix((n, n), dtype=bool)

    accumated = copy.deepcopy(productions)

    for n, m, tag in graph.edges.data("label"):
        if tag in terminals:
            for terminalN in terminals[tag]:
                productions[terminalN][n, m] = True

    for eps in epsilons:
        productions[eps].setdiag(True)

    newVals = True
    while newVals:
        newVals = False
        for multN, mult in mults.items():
            prev = accumated[multN].nnz
            for l, r in mult:
                accumated[multN] += productions[l] @ productions[r]

            newVals |= prev != accumated[multN].nnz
        if newVals:
            for accN, m in accumated.items():
                productions[accN] += m

    start = cfg.start_symbol.value

    return {
        (n, m)
        for n, m in zip(*productions[start].nonzero())
        if n in start_nodes and m in final_nodes
    }

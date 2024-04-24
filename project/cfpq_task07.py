import networkx as nx
import pyformlang
import scipy.sparse
from pyformlang.cfg import Variable, Epsilon
from scipy.sparse import dok_matrix, csr_matrix

from project.onfh_task06 import cfg_to_weak_normal_form


def cfpq_with_matrix(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:

    cfg_weak = cfg_to_weak_normal_form(cfg)
    nodes_len = len(graph.nodes)

    pre_res = {}

    ni_to_nj_nk = set()
    for v in cfg_weak.variables:
        pre_res[v] = dok_matrix((nodes_len, nodes_len), dtype=bool)

    for i, j, tag in graph.edges.data("label"):
        for prods in cfg_weak.productions:
            if (
                len(prods.body) == 1
                and isinstance(prods.body[0], Variable)
                and prods.body[0].value == tag
            ):
                pre_res[prods.head][i, j] = True
            elif len(prods.body) == 1 and isinstance(prods.body[0], Epsilon):
                pre_res[prods.head] += csr_matrix(
                    scipy.sparse.eye(nodes_len), dtype=bool
                )
            elif len(prods.body) == 2:
                ni_to_nj_nk.add((prods.head, prods.body[0], prods.body[1]))
            else:
                pass  # ?????

    allNodes = {}
    for i, node in enumerate(graph.nodes):
        allNodes[i] = node

    res_csrs = {}
    for x, matrix in pre_res.items():
        res_csrs[x] = csr_matrix(matrix)

    notChanged = False
    while not notChanged:
        notChanged = True  # hack for empty ni_to_nj_nk
        for ni, nj, nk in ni_to_nj_nk:
            prev = res_csrs[ni].nnz
            res_csrs[ni] += res_csrs[nj] @ res_csrs[nk]
            if prev == res_csrs[ni].nnz:
                notChanged = True
            else:
                notChanged = False

    result = set()
    for k, matrix in res_csrs.items():
        for i, j in zip(*matrix.nonzero()):
            result.add((allNodes[i], allNodes[j]))

    return result

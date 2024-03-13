from project.finite_automaton_task03 import (
    FiniteAutomaton,
    intersect_automata,
    make_transitive_closure,
)


def reachability_with_constraints(
    fa: FiniteAutomaton, constraints_fa: FiniteAutomaton
) -> dict[int, set[int]]:

    intersected = intersect_automata(fa, constraints_fa)

    transitive_closure = make_transitive_closure(intersected)

    fa_mapping = {v: i for i, v in fa.mapping.items()}

    result = dict()
    for s in fa.start:
        result[s] = set()

    fa_len = len(constraints_fa.mapping)
    for u, v in zip(*transitive_closure.nonzero()):
        if u in intersected.start and v in intersected.final:
            result[fa_mapping[u // fa_len]].add(fa_mapping[v // fa_len])

    return result

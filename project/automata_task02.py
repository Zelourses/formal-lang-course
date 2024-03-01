from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from pyformlang.regular_expression import Regex
from networkx import MultiDiGraph


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    regex = Regex(regex)
    automata = regex.to_epsilon_nfa()
    res = automata.to_deterministic()
    return res.minimize()


def graph_to_nfa(
    graph: MultiDiGraph, start_states: set[int], final_states: set[int]
) -> NondeterministicFiniteAutomaton:
    allStates = set(graph.nodes)
    start = (
        allStates if start_states is None or len(start_states) == 0 else start_states
    )
    final = (
        allStates if final_states is None or len(final_states) == 0 else final_states
    )

    nfa = NondeterministicFiniteAutomaton()
    for s in start:
        nfa.add_start_state(State(s))
    for s in final:
        nfa.add_final_state(State(s))

    for u, v, edgeData in graph.edges(data=True):
        nfa.add_transition(
            s_from=State(u), symb_by=Symbol(edgeData["label"]), s_to=State(v)
        )

    return nfa

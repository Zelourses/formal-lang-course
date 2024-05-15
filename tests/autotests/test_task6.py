# This file contains test cases that you need to pass to get a grade
# You MUST NOT touch anything here except ONE block below
# You CAN modify this file IF AND ONLY IF you have found a bug and are willing to fix it
# Otherwise, please report it
import pytest
from grammars_constants import REGEXP_CFG, GRAMMARS
from rpq_template_test import rpq_cfpq_test, different_grammars_test
from fixtures import graph

# Fix import statements in try block to run tests
try:
    from project.automata_task02 import graph_to_nfa, regex_to_dfa
    from project.finite_automaton_task03 import FiniteAutomaton
    from project.reachability_task04 import reachability_with_constraints
    from project.onfh_task06 import cfpq_with_hellings
except ImportError:
    pytestmark = pytest.mark.skip("Task 6 is not ready to test!")


class TestReachability:
    @pytest.mark.parametrize("regex_str, cfg_list", REGEXP_CFG)
    def test_rpq_cfpq_hellings(self, graph, regex_str, cfg_list):
        rpq_cfpq_test(graph, regex_str, cfg_list, cfpq_with_hellings)

    @pytest.mark.parametrize("eq_grammars", GRAMMARS)
    def test_different_grammars_hellings(self, graph, eq_grammars):
        different_grammars_test(graph, eq_grammars, cfpq_with_hellings)


def test_cfg_to_weak_normal_form_exists():
    try:
        import project.onfh_task06

        assert "cfg_to_weak_normal_form" in dir(project.onfh_task06)
    except NameError:
        assert False

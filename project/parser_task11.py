from project.graphminusminus.graphminusminusLexer import graphminusminusLexer
from project.graphminusminus.graphminusminusListener import graphminusminusListener
from project.graphminusminus.graphminusminusParser import graphminusminusParser

from antlr4 import Parser, ParserRuleContext, CommonTokenStream
from antlr4.InputStream import InputStream


class NodeCounter(graphminusminusListener):
    counter: int = 0

    def __init__(self) -> None:
        super().__init__()

    def enterEveryRule(self, ctx: ParserRuleContext) -> None:
        self.counter += 1


class NodeStringify(graphminusminusListener):
    resString: str = ""

    def __init__(self) -> None:
        super().__init__()

    def enterEveryRule(self, ctx: ParserRuleContext) -> None:
        self.resString += ctx.getText()


# Второе поле показывает корректна ли строка (True, если корректна)
def prog_to_tree(prog: str) -> tuple[ParserRuleContext, bool]:
    parser = graphminusminusParser(
        CommonTokenStream(graphminusminusLexer(InputStream(prog)))
    )
    prog: ParserRuleContext = parser.prog()
    return prog, parser.getNumberOfSyntaxErrors() == 0


def nodes_count(tree: ParserRuleContext) -> int:
    listener = NodeCounter()
    tree.enterRule(listener)
    return listener.counter


def tree_to_prog(tree: ParserRuleContext) -> str:
    listener = NodeStringify()
    tree.enterRule(listener)
    return listener.resString

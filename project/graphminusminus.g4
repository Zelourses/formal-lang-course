grammar graphminusminus;

prog: stmt* EOF;

stmt: bind | add | remove | declare;

declare: 'let' VAR 'is' 'graph';

bind: 'let' VAR '=' expr;

remove
    : 'remove' ('vertex' | 'edge' | 'vertices') expr 'from' VAR;

add: 'add' ('vertex' | 'edge') expr 'to' VAR;

expr: NUM | CHAR | VAR | edgeExpr | setExpr | regexp | select;

setExpr: '[' expr (',' expr)* ']';

edgeExpr: '(' expr ',' expr ',' expr ')';

regexp
    : CHAR
    | VAR
    | '(' regexp ')'
    | regexp '|' regexp
    | regexp '^' range
    | regexp '.' regexp
    | regexp '&' regexp;

range: '[' NUM '..' NUM? ']';

select
    : vFilter? vFilter? 'return' VAR (',' VAR)? 'where' VAR 'reachable' 'from' VAR 'in' VAR 'by'
        expr;

vFilter: 'for' VAR 'in' expr;

VAR: [a-z][a-z0-9]*;
NUM: '0' | ([1-9][0-9]*);
CHAR: '"' [a-z] '"';

WS: [ \r\n\t]+ -> skip;

"""T-SQL to DuckDB translation, identical to legacy/sqlshare_translate.transpile_tsql."""
from __future__ import annotations

import re

PATIDX = re.compile(r"\bPATINDEX\s*\(", re.I)


def transpile_tsql(sql: str) -> str:
    """Transpile T-SQL to DuckDB via sqlglot, repairing string concatenation.

    T-SQL overloads ``+`` as string concatenation. Where one operand is a string literal
    (or an already rewritten concatenation) the node is rewritten to ``||``.
    """
    import sqlglot
    from sqlglot import exp

    tree = sqlglot.parse_one(sql, read="tsql")

    def is_stringy(node) -> bool:
        if isinstance(node, exp.Literal):
            return bool(node.is_string)
        return isinstance(node, exp.DPipe)

    def fix_concat(node):
        if isinstance(node, exp.Add) and (is_stringy(node.left) or is_stringy(node.right)):
            return exp.DPipe(this=node.left, expression=node.right)
        return node

    prev = None
    for _ in range(10):
        cur = tree.sql()
        if cur == prev:
            break
        prev = cur
        tree = tree.transform(fix_concat)
    return tree.sql(dialect="duckdb")


def classify_error(msg: str) -> str:
    m = msg.lower()
    if "does not have a column named" in m or "referenced column" in m or "not found in from" in m:
        return "column_not_found"
    if "table with name" in m or "does not exist" in m or "catalog error" in m:
        return "relation_not_found"
    if "parser error" in m or "syntax error" in m:
        return "parse_error"
    if "binder error" in m:
        return "binder_other"
    if "conversion" in m or "could not convert" in m or "cast" in m:
        return "type_error"
    if "interrupt" in m:
        return "timeout"
    return "other"

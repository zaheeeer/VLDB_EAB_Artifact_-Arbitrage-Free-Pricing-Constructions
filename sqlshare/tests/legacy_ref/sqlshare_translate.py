"""Translate SQLShare's SQL Server dialect into DuckDB, and report why queries fail.

The translation is deliberately conservative. Each rule is a narrow, reversible
rewrite, and the set of rules that fired is recorded per query so translation attrition
can be attributed to a cause instead of reported as a bare count.

Constructs with no safe regex rewrite (``TOP n PERCENT``, ``IIF``, temp tables, ``PIVOT``,
three-argument ``CONVERT``) are detected and the query is marked unsupported rather than
mangled into something that parses but means something else.
"""
from __future__ import annotations

import re

UNSUPPORTED = {
    "top_percent": re.compile(r"\bTOP\s*\(?\s*\d+\s*\)?\s+PERCENT\b", re.I),
    "iif": re.compile(r"\bIIF\s*\(", re.I),
    "temp_table": re.compile(r"(^|[\s(,])#\w+", re.M),
    "pivot": re.compile(r"\b(PIVOT|UNPIVOT)\b", re.I),
    "convert_style": re.compile(r"\bCONVERT\s*\([^,()]+,[^,()]+,[^,()]+\)", re.I),
    "declare": re.compile(r"\bDECLARE\s+@", re.I),
    "cursor": re.compile(r"\b(CURSOR|EXEC|EXECUTE)\b", re.I),
    "into_table": re.compile(r"\bINTO\s+[\[#\w]", re.I),
}

BRACKET_PAIR = re.compile(r"\[([^\[\]]+)\]\s*\.\s*\[([^\[\]]+)\]")
BRACKET_ONE = re.compile(r"\[([^\[\]]+)\]")
TOP_N = re.compile(r"\bSELECT\s+(DISTINCT\s+)?TOP\s*\(?\s*(\d+)\s*\)?\s+", re.I)
CONVERT2 = re.compile(r"\bCONVERT\s*\(\s*([A-Za-z0-9_]+(?:\s*\(\s*\d+\s*\))?)\s*,\s*", re.I)
DATEPART = re.compile(r"\bDATEPART\s*\(\s*([A-Za-z]+)\s*,", re.I)

SIMPLE = [
    ("isnull", re.compile(r"\bISNULL\s*\(", re.I), "COALESCE("),
    ("len", re.compile(r"\bLEN\s*\(", re.I), "LENGTH("),
    ("getdate", re.compile(r"\bGETDATE\s*\(\s*\)", re.I), "current_timestamp"),
    ("square_root", re.compile(r"\bSQUARE\s*\(", re.I), "pow2_placeholder("),
]


def transpile_tsql(sql: str) -> str:
    """Transpile T-SQL to DuckDB via sqlglot, repairing string concatenation.

    T-SQL overloads ``+`` as string concatenation. sqlglot leaves it as arithmetic when
    it cannot infer operand types, and DuckDB then rejects ``'a' + varchar``. Where one
    operand is a string literal the intent is unambiguous, so the node is rewritten to
    ``||``. Numeric additions are untouched.
    """
    import sqlglot
    from sqlglot import exp

    tree = sqlglot.parse_one(sql, read="tsql")

    def is_stringy(node) -> bool:
        if isinstance(node, exp.Literal):
            return bool(node.is_string)
        # A concatenation already rewritten is string-valued, which lets nested
        # ``'a' + col + '-'`` chains converge instead of converting only the outer node.
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


def find_unsupported(sql: str):
    return sorted(k for k, rx in UNSUPPORTED.items() if rx.search(sql))


def translate(sql: str):
    """Return (translated_sql, rules_fired, unsupported_reasons)."""
    bad = find_unsupported(sql)
    if bad:
        return None, [], bad

    fired = []
    out = sql.strip().rstrip(";")

    # TOP n -> LIMIT n. Must run before bracket rewriting so the SELECT prefix is intact.
    limit = None
    m = TOP_N.search(out)
    if m:
        limit = int(m.group(2))
        out = TOP_N.sub(lambda mm: "SELECT " + (mm.group(1) or ""), out, count=1)
        fired.append("top_to_limit")

    # CONVERT(type, expr) -> CAST(expr AS type). Two-argument form only.
    def conv(mm):
        fired.append("convert_to_cast")
        return f"CAST("
    if CONVERT2.search(out):
        def repl(mm):
            return "CAST("
        # rewrite head, then patch the closing by inserting the AS clause
        def convert_once(s):
            mm = CONVERT2.search(s)
            if not mm:
                return s, False
            typ = mm.group(1)
            start = mm.start()
            i = mm.end()
            depth = 1
            while i < len(s) and depth:
                if s[i] == "(":
                    depth += 1
                elif s[i] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            if depth:
                return s, False
            inner = s[mm.end():i]
            return s[:start] + f"CAST({inner} AS {typ})" + s[i + 1:], True
        changed = True
        n = 0
        while changed and n < 20:
            out, changed = convert_once(out)
            n += changed
        if n:
            fired.append("convert_to_cast")

    # DATEPART(part, x) -> date_part('part', x)
    if DATEPART.search(out):
        out = DATEPART.sub(lambda mm: f"date_part('{mm.group(1).lower()}',", out)
        fired.append("datepart")

    for name, rx, rep in SIMPLE:
        if rx.search(out):
            out = rx.sub(rep, out)
            fired.append(name)

    # Bracketed identifiers -> double-quoted identifiers. Pairs first so the dot survives.
    if BRACKET_PAIR.search(out):
        out = BRACKET_PAIR.sub(lambda mm: f'"{mm.group(1)}"."{mm.group(2)}"', out)
        fired.append("qualified_ident")
    if BRACKET_ONE.search(out):
        out = BRACKET_ONE.sub(lambda mm: f'"{mm.group(1)}"', out)
        fired.append("bare_ident")

    if limit is not None:
        out = f"{out} LIMIT {limit}"

    return out, sorted(set(fired)), []


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
    return "other"

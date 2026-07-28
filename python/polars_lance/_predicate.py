"""Translation of Polars predicates into Lance SQL filter strings.

Polars' optimizer decides *which* predicate belongs to a scan: it pushes filters down
through joins and splits conjunctions by column dependency, so the predicate handed to an
IO plugin only references columns that scan produces. It has no knowledge of Lance's SQL
dialect though, so that predicate may still contain expressions Lance cannot evaluate.

`to_lance_sql` translates the parts that Lance *can* evaluate, and returns `None` when
nothing can be translated. The result is only used to reduce the rows Lance reads: the
caller re-applies the full predicate to every batch, so a translation is only allowed to
match a *superset* of the rows the predicate selects, never a subset. Anything that cannot
be translated with that guarantee is left out.

Polars and SQL do not agree on everything, and a translation that looks right can quietly
select the wrong rows, so every construct here is checked against both engines on data
containing nulls, empty strings and lists, null structs, and regex metacharacters. See
`tests/test_predicate_equivalence.py`; do not add a construct without extending it.

Three differences found that way are worth stating, because they are not obvious:

- Whether Lance records the validity of a *struct* itself depends on the file format
  version: a null struct survives a round trip in version 2.1 but not in 2.0, where it
  comes back as a valid struct whose fields carry the writer's filler values. Worse, the
  filter and the scan can disagree about the same row, so a pushed `IS NULL` may select a
  row that then materializes as non-null and is dropped again by the caller's filter, which
  would lose it (https://github.com/lance-format/lance/issues/7908). `IS NULL` is therefore
  not translated for a struct column, nor for a field read out of a struct. Comparisons on
  a struct field are translated: they agree in both versions.
- Polars' `list.get` raises on an out-of-bounds index while SQL yields null, and Lance
  indexes lists from 1 where Polars indexes from 0, so list element access is not
  translated at all. `list.len` and `list.contains` are.
- Polars compares `NaN` as an ordinary value that is equal to itself, which SQL does not,
  so a comparison against a `NaN` literal is not translated.

The predicate is inspected through its serialized form, which is not stable across Polars
versions. An unrecognized expression only means the predicate is not pushed down, so a
change to that format costs performance, never correctness.
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import polars as pl

_EPOCH_DATE = date(1970, 1, 1)

_COMPARISON_OPERATORS = {
    "Eq": "=",
    "NotEq": "!=",
    "Lt": "<",
    "LtEq": "<=",
    "Gt": ">",
    "GtEq": ">=",
}

# `EqValidity` and `NotEqValidity` treat nulls as comparable values, which SQL's `=` and
# `!=` do not, so they are deliberately absent.
_CONJUNCTION_OPERATORS = frozenset({"And", "LogicalAnd"})
_DISJUNCTION_OPERATORS = frozenset({"Or", "LogicalOr"})

_BETWEEN_OPERATORS = {
    "Both": (">=", "<="),
    "Left": (">=", "<"),
    "Right": (">", "<="),
    "None": (">", "<"),
}

Schema = Mapping[str, Any]


@dataclass(frozen=True)
class _Operand:
    """A translated value expression, with what is known about the value it produces."""

    sql: str
    dtype: Any | None = None
    # Whether the value is read out of a struct. Lance evaluates such a read against the
    # leaf field only, so it cannot tell a null field from a field of a null struct.
    from_struct_field: bool = False


def to_lance_sql(predicate: pl.Expr, schema: Schema | None = None) -> str | None:
    """
    Translate a Polars predicate into a Lance SQL filter string.

    Parameters
    ----------
    predicate
        The predicate Polars pushed into the scan.
    schema
        Dtypes of the columns the scan produces. Without it, constructs whose soundness
        depends on the column's dtype are not translated.

    Returns
    -------
    The filter string, or `None` if no part of the predicate can be safely translated.
    """
    try:
        tree = json.loads(predicate.meta.serialize(format="json"))
    except Exception:
        return None

    return _expression_to_sql(tree, schema or {})


def _expression_to_sql(node: Any, schema: Schema) -> str | None:
    if not isinstance(node, dict):
        return None

    if "BinaryExpr" in node:
        return _binary_expression_to_sql(node["BinaryExpr"], schema)

    if "Function" in node:
        return _function_to_sql(node["Function"], schema)

    return None


def _binary_expression_to_sql(node: Any, schema: Schema) -> str | None:
    if not isinstance(node, dict):
        return None

    operator = node.get("op")
    left, right = node.get("left"), node.get("right")

    # A conjunction can be weakened: if only one side translates, pushing just that side
    # still matches a superset of the rows the full predicate selects.
    if operator in _CONJUNCTION_OPERATORS:
        left_sql = _expression_to_sql(left, schema)
        right_sql = _expression_to_sql(right, schema)
        if left_sql is not None and right_sql is not None:
            return f"({left_sql} AND {right_sql})"
        return left_sql if left_sql is not None else right_sql

    # A disjunction cannot be weakened: dropping a side would exclude rows that only that
    # side matches, yielding a subset.
    if operator in _DISJUNCTION_OPERATORS:
        left_sql = _expression_to_sql(left, schema)
        right_sql = _expression_to_sql(right, schema)
        if left_sql is None or right_sql is None:
            return None
        return f"({left_sql} OR {right_sql})"

    if not isinstance(operator, str):
        return None

    sql_operator = _COMPARISON_OPERATORS.get(operator)
    if sql_operator is None:
        return None

    left_operand = _operand_to_sql(left, schema)
    right_operand = _operand_to_sql(right, schema)
    if left_operand is None or right_operand is None:
        return None

    # Polars compares NaN as an ordinary value that equals itself, which SQL does not do.
    if _is_nan_literal(left_operand) or _is_nan_literal(right_operand):
        return None

    return f"({left_operand.sql} {sql_operator} {right_operand.sql})"


def _is_nan_literal(operand: _Operand) -> bool:
    return operand.sql == _NAN_SQL


def _function_to_sql(node: Any, schema: Schema) -> str | None:
    """Translate a function that produces a boolean, i.e. one usable as a filter."""
    if not isinstance(node, dict):
        return None

    function = node.get("function")
    inputs = node.get("input")
    if not isinstance(inputs, list) or not inputs:
        return None

    if not isinstance(function, dict):
        return None

    if "Boolean" in function:
        return _boolean_function_to_sql(function["Boolean"], inputs, schema)

    if "StringExpr" in function:
        return _string_predicate_to_sql(function["StringExpr"], inputs, schema)

    if "ListExpr" in function:
        return _list_predicate_to_sql(function["ListExpr"], inputs, schema)

    return None


def _boolean_function_to_sql(
    boolean: Any, inputs: list[Any], schema: Schema
) -> str | None:
    if boolean in ("IsNull", "IsNotNull"):
        operand = _operand_to_sql(inputs[0], schema)
        if operand is None or not _null_check_is_sound(operand):
            return None
        predicate = "IS NULL" if boolean == "IsNull" else "IS NOT NULL"
        return f"({operand.sql} {predicate})"

    if not isinstance(boolean, dict):
        # `Not` is not translated: negating a weakened translation would turn a superset
        # into a subset.
        return None

    if "IsIn" in boolean:
        # `nulls_equal` makes a null match a null, which SQL's `IN` does not do.
        if (boolean["IsIn"] or {}).get("nulls_equal"):
            return None
        return _is_in_to_sql(inputs, schema)

    if "IsBetween" in boolean:
        closed = (boolean["IsBetween"] or {}).get("closed")
        return _is_between_to_sql(inputs, closed, schema)

    return None


def _null_check_is_sound(operand: _Operand) -> bool:
    """
    Whether `IS NULL` on this operand agrees with Polars.

    Struct validity is not reliable to filter on: whether it survives a write depends on the
    file format version, and the filter and the scan can disagree about the same row.
    """
    if operand.from_struct_field:
        return False

    if operand.dtype is None:
        return False

    # `Array` is excluded because its null handling has not been checked against Polars.
    return not isinstance(operand.dtype, (pl.Struct, pl.Array))


def _is_in_to_sql(inputs: list[Any], schema: Schema) -> str | None:
    if len(inputs) != 2:
        return None

    operand = _operand_to_sql(inputs[0], schema)
    if operand is None:
        return None

    values = _list_literal_values(inputs[1])
    if not values:
        return None

    rendered = []
    for value in values:
        # A null cannot be matched by SQL's `IN`, so such a filter would match a subset.
        if value is None:
            return None
        rendered_value = _python_value_to_sql(value)
        if rendered_value is None:
            return None
        rendered.append(rendered_value)

    return f"({operand.sql} IN ({', '.join(rendered)}))"


def _is_between_to_sql(inputs: list[Any], closed: Any, schema: Schema) -> str | None:
    if len(inputs) != 3:
        return None

    operators = _BETWEEN_OPERATORS.get(closed)
    if operators is None:
        return None
    lower_operator, upper_operator = operators

    operand = _operand_to_sql(inputs[0], schema)
    lower = _operand_to_sql(inputs[1], schema)
    upper = _operand_to_sql(inputs[2], schema)
    if operand is None or lower is None or upper is None:
        return None

    return (
        f"({operand.sql} {lower_operator} {lower.sql}"
        f" AND {operand.sql} {upper_operator} {upper.sql})"
    )


def _string_predicate_to_sql(
    string_expr: Any, inputs: list[Any], schema: Schema
) -> str | None:
    if len(inputs) != 2:
        return None

    operand = _operand_to_sql(inputs[0], schema)
    pattern = _operand_to_sql(inputs[1], schema)
    if operand is None or pattern is None:
        return None

    if string_expr == "StartsWith":
        return f"starts_with({operand.sql}, {pattern.sql})"

    if string_expr == "EndsWith":
        return f"ends_with({operand.sql}, {pattern.sql})"

    if isinstance(string_expr, dict) and "Contains" in string_expr:
        options = string_expr["Contains"] or {}
        if options.get("literal"):
            return f"contains({operand.sql}, {pattern.sql})"
        # Both engines use the Rust regex crate, so the pattern needs no rewriting.
        return f"regexp_like({operand.sql}, {pattern.sql})"

    return None


def _list_predicate_to_sql(
    list_expr: Any, inputs: list[Any], schema: Schema
) -> str | None:
    if not isinstance(list_expr, dict) or "Contains" not in list_expr:
        return None

    if len(inputs) != 2:
        return None

    operand = _operand_to_sql(inputs[0], schema)
    value = _operand_to_sql(inputs[1], schema)
    if operand is None or value is None:
        return None

    return f"array_has({operand.sql}, {value.sql})"


def _operand_to_sql(node: Any, schema: Schema) -> _Operand | None:
    """Translate an expression that produces a value rather than a filter."""
    if not isinstance(node, dict):
        return None

    if "Column" in node:
        return _column_to_sql(node["Column"], schema)

    if "Literal" in node:
        sql = _literal_to_sql(node["Literal"])
        return None if sql is None else _Operand(sql)

    if "Function" in node:
        return _value_function_to_sql(node["Function"], schema)

    return None


def _value_function_to_sql(node: Any, schema: Schema) -> _Operand | None:
    if not isinstance(node, dict):
        return None

    function = node.get("function")
    inputs = node.get("input")
    if not isinstance(function, dict) or not isinstance(inputs, list) or not inputs:
        return None

    inner = _operand_to_sql(inputs[0], schema)
    if inner is None:
        return None

    if "StructExpr" in function:
        struct_expr = function["StructExpr"]
        if not isinstance(struct_expr, dict) or "FieldByName" not in struct_expr:
            return None
        return _struct_field_to_sql(inner, struct_expr["FieldByName"])

    if "StringExpr" in function and len(inputs) == 1:
        string_expr = function["StringExpr"]
        if string_expr == "Lowercase":
            return _Operand(f"lower({inner.sql})", pl.String, inner.from_struct_field)
        if string_expr == "Uppercase":
            return _Operand(f"upper({inner.sql})", pl.String, inner.from_struct_field)
        if string_expr == "LenChars":
            return _Operand(
                f"character_length({inner.sql})", pl.UInt32, inner.from_struct_field
            )
        return None

    if "ListExpr" in function and function["ListExpr"] == "Length" and len(inputs) == 1:
        return _Operand(
            f"array_length({inner.sql})", pl.UInt32, inner.from_struct_field
        )

    return None


def _struct_field_to_sql(inner: _Operand, field: Any) -> _Operand | None:
    if not isinstance(field, str) or "." in field or "`" in field:
        return None

    dtype = None
    if isinstance(inner.dtype, pl.Struct):
        dtype = {f.name: f.dtype for f in inner.dtype.fields}.get(field)
        if dtype is None:
            return None

    return _Operand(f"{inner.sql}.`{field}`", dtype, from_struct_field=True)


def _column_to_sql(name: Any, schema: Schema) -> _Operand | None:
    if not isinstance(name, str):
        return None

    # Lance rejects a top level field name containing a period when the dataset is
    # written, so such a column cannot be scanned. Guarded anyway, because emitting it
    # would read as a struct field access.
    if "." in name:
        return None

    escaped = name.replace("`", "``")
    return _Operand(f"`{escaped}`", schema.get(name))


def _literal_to_sql(node: Any) -> str | None:
    if not isinstance(node, dict):
        return None

    scalar = node.get("Scalar")
    if isinstance(scalar, dict) and len(scalar) == 1:
        ((dtype, value),) = scalar.items()
        return _scalar_to_sql(dtype, value)

    # A literal whose dtype the optimizer has not resolved yet. Polars resolves literals
    # before handing a predicate to an IO plugin, but an expression that is translated
    # without being optimized still carries them.
    dynamic = node.get("Dyn")
    if isinstance(dynamic, dict) and len(dynamic) == 1:
        ((dtype, value),) = dynamic.items()
        return _dynamic_literal_to_sql(dtype, value)

    # `Series` and `Range` literals are not scalars.
    return None


def _dynamic_literal_to_sql(dtype: str, value: Any) -> str | None:
    if dtype == "Int":
        return str(value) if isinstance(value, int) else None

    if dtype == "Float":
        return _float_to_sql(value)

    if dtype == "Str":
        return _string_to_sql(value) if isinstance(value, str) else None

    return None


def _scalar_to_sql(dtype: str, value: Any) -> str | None:
    if dtype == "Boolean":
        return "true" if value else "false"

    if dtype.startswith("Int") or dtype.startswith("UInt"):
        return str(value) if isinstance(value, int) else None

    if dtype.startswith("Float"):
        return _float_to_sql(value)

    if dtype == "String":
        return _string_to_sql(value) if isinstance(value, str) else None

    if dtype == "Date":
        return _date_to_sql(value)

    if dtype == "Datetime":
        return _datetime_to_sql(value)

    # `Null` has no SQL literal (`IS NULL` is used instead), and time, duration, decimal,
    # and binary literals are not translated.
    return None


# Rendered for a NaN literal so a comparison against one can be rejected: Polars treats
# NaN as a value equal to itself, SQL does not.
_NAN_SQL = "<nan>"


def _float_to_sql(value: Any) -> str | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None

    if math.isnan(value):
        return _NAN_SQL

    # Infinities have no SQL literal representation.
    if math.isinf(value):
        return None

    # `repr` always renders a decimal point, so the literal is not read as an integer.
    return repr(float(value))


def _string_to_sql(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _date_to_sql(days_since_epoch: Any) -> str | None:
    if not isinstance(days_since_epoch, int) or isinstance(days_since_epoch, bool):
        return None

    try:
        value = _EPOCH_DATE + timedelta(days=days_since_epoch)
    except OverflowError:
        return None

    return f"date '{value.isoformat()}'"


def _datetime_to_sql(value: Any) -> str | None:
    if not isinstance(value, list) or len(value) != 3:
        return None

    amount, time_unit, time_zone = value

    # Time zone aware timestamps are not translated, to avoid an off-by-offset filter.
    if time_zone is not None:
        return None

    if not isinstance(amount, int) or isinstance(amount, bool):
        return None

    if time_unit == "Milliseconds":
        microseconds = amount * 1_000
    elif time_unit == "Microseconds":
        microseconds = amount
    elif time_unit == "Nanoseconds":
        # A SQL timestamp literal has microsecond precision. Truncating would shift the
        # comparison, so sub-microsecond values are not translated.
        if amount % 1_000 != 0:
            return None
        microseconds = amount // 1_000
    else:
        return None

    try:
        moment = datetime.fromtimestamp(microseconds / 1_000_000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None

    return f"timestamp '{moment.strftime('%Y-%m-%d %H:%M:%S.%f')}'"


def _list_literal_values(node: Any) -> list[Any] | None:
    """Read the values of a list literal, which is serialized as Arrow IPC bytes."""
    scalar = node.get("Literal", {}).get("Scalar") if isinstance(node, dict) else None
    if not isinstance(scalar, dict):
        return None

    data = scalar.get("List")
    if not isinstance(data, list):
        return None

    try:
        frame = pl.read_ipc_stream(io.BytesIO(bytes(data)))
    except Exception:
        return None

    if frame.width != 1:
        return None

    return frame.to_series().to_list()


def _python_value_to_sql(value: Any) -> str | None:
    """Render a value read from a list literal."""
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        rendered = _float_to_sql(value)
        return None if rendered == _NAN_SQL else rendered

    if isinstance(value, str):
        return _string_to_sql(value)

    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return None
        return f"timestamp '{value.strftime('%Y-%m-%d %H:%M:%S.%f')}'"

    if isinstance(value, date):
        return f"date '{value.isoformat()}'"

    return None

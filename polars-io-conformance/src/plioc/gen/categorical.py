"""Categorical and Enum.

The two are different contracts and fail differently. An `Enum` carries its categories in the
dtype, in a fixed order, including categories no row uses -- all of which a writer can drop. A
`Categorical` carries only the values; its physical codes are an implementation detail that a
round-trip is free to renumber, which is why comparing it exactly is a `PHYSICAL`-level claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from plioc.gen.core import GenContext, pick

#: Deliberately not sorted, and not in first-use order: a writer that rebuilds the category list
#: from the data recovers a different order, and only a case like this notices.
DEFAULT_CATEGORIES: tuple[str, ...] = ("zeta", "alpha", "", "Ω", "alpha ", "beta")


@dataclass(frozen=True)
class EnumGen:
    categories: tuple[str, ...] = DEFAULT_CATEGORIES
    #: Categories present in the dtype that no row ever takes. Lost by any writer that infers
    #: the category set from the values.
    unused: tuple[str, ...] = ("never-used",)

    @property
    def dtype(self) -> pl.DataType:
        return pl.Enum(list(self.categories) + list(self.unused))

    def expr(self, ctx: GenContext) -> pl.Expr:
        pal = pl.Series("palette", list(self.categories), dtype=self.dtype)
        return pick(ctx, pal)


@dataclass(frozen=True)
class CategoricalGen:
    values: tuple[str, ...] = DEFAULT_CATEGORIES

    @property
    def dtype(self) -> pl.DataType:
        return pl.Categorical()

    def expr(self, ctx: GenContext) -> pl.Expr:
        pal = pl.Series("palette", list(self.values), dtype=pl.Categorical())
        return pick(ctx, pal)


@dataclass(frozen=True)
class EmptyEnumGen:
    """An `Enum` with no categories at all, so every row is null. Legal, and rare enough that
    writers mishandle the zero-width dictionary."""

    _unused: int = field(default=0, repr=False)

    @property
    def dtype(self) -> pl.DataType:
        return pl.Enum([])

    def expr(self, ctx: GenContext) -> pl.Expr:
        return pl.repeat(None, pl.len(), dtype=self.dtype)

"""Shared progress helpers for formation experiment runners."""

from __future__ import annotations

from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


class _NullProgress:
    """Fallback shim when tqdm is unavailable."""

    def __init__(
        self,
        iterable: Optional[Iterable[T]] = None,
        *,
        total: Optional[int] = None,
        desc: str = "",
        **_: object,
    ) -> None:
        self.iterable = iterable
        self.total = total
        self.desc = desc
        self.n = 0

    def __iter__(self) -> Iterator[T]:
        if self.iterable is None:
            return
        for item in self.iterable:
            self.n += 1
            yield item

    def update(self, n: int = 1) -> None:
        self.n += int(n)

    def set_postfix(self, *args, **kwargs) -> None:
        return None

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        del refresh
        if desc is not None:
            self.desc = str(desc)

    def write(self, message: str) -> None:
        print(message)

    def close(self) -> None:
        return None


def create_progress(
    *,
    iterable: Optional[Iterable[T]] = None,
    total: Optional[int] = None,
    desc: str = "",
    unit: str = "it",
    leave: bool = True,
    disable: bool = False,
    position: Optional[int] = None,
):
    kwargs = {
        "total": total,
        "desc": desc,
        "unit": unit,
        "leave": leave,
        "disable": disable,
        "dynamic_ncols": True,
        "mininterval": 0.3,
    }
    if position is not None:
        kwargs["position"] = position
    if _tqdm is None:
        return _NullProgress(iterable=iterable, **kwargs)
    if iterable is None:
        return _tqdm(**kwargs)
    return _tqdm(iterable, **kwargs)


def progress_range(
    stop: int,
    *,
    desc: str,
    unit: str = "step",
    leave: bool = True,
    disable: bool = False,
    position: Optional[int] = None,
):
    total = max(0, int(stop))
    return create_progress(
        iterable=range(total),
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        disable=disable,
        position=position,
    )


def progress_iter(
    iterable: Iterable[T],
    *,
    total: Optional[int] = None,
    desc: str,
    unit: str = "it",
    leave: bool = True,
    disable: bool = False,
    position: Optional[int] = None,
):
    return create_progress(
        iterable=iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        disable=disable,
        position=position,
    )

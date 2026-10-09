# wavelet_tree

A wavelet tree over an integer alphabet. Builds once from a sequence of ints and answers **rank**, **select**, and **range-quantile** queries in `O(alphabet_bits)` time, using standard library only.

```python
from wavelet_tree import WaveletTree

wt = WaveletTree([3, 1, 4, 1, 5, 9, 2, 6])

wt.rank(1)                 # -> 2  (count of 1 in the whole sequence)
wt.rank(1, 5)              # -> 2  (count of 1 in wt[:5])
wt.select(1, 2)            # -> 3  (0-based index of the 2nd occurrence of 1)
wt.range_quantile(2, 6, 1) # -> 4  (2nd smallest in wt[2:6] == [4,1,5,9])
```

## Why this exists

Some workloads need ranked queries over a static sequence — count how many times a symbol appeared up to position *p*, find the *k*-th occurrence of a symbol, or pull the *k*-th smallest value inside a sub-range — without paying for a full segment tree per value. A wavelet tree does all three in `O(log A)` per query, where `A` is the alphabet width, while keeping memory at roughly `n log A` bits. The trade-off versus a sorted index is that you must build the whole tree up front and it is immutable after that.

Symbols are any Python `int` (negative values included). `bool` is rejected explicitly because it subclasses `int` and silently mixing `True`/`False` with real integers is the kind of surprise this library is meant to avoid.

## The awkward edges

- `select` is 1-based in its `k` argument ("find the *k*-th occurrence"), but the returned index is 0-based. Both choices are documented in the method's docstring; the asymmetry is deliberate so `select(s, 1)` always means "first occurrence."
- `range_quantile`'s `k` is 0-based, matching the "k-th smallest" convention from statistics. Mixing 1-based `select` and 0-based `range_quantile` is the one place callers trip.
- The alphabet is the *compact* range `[min_symbol, max_symbol]` from the input. A symbol inside that range but not present in the sequence returns `0` from `rank` and raises `ValueError` from `select`; symbols outside the range behave the same way. There is no separate "alphabet registration" step.
- An empty sequence builds cleanly; every query on it raises `ValueError` rather than returning a silent zero, because a 0 result on an empty tree would collide with the legitimate "symbol not present" case.

## API

The single exported class is `WaveletTree`, constructed from an iterable of `int`.

- `len(wt)` — sequence length.
- `wt.rank(symbol, upto=None)` — occurrences of `symbol` in `wt[:upto]`. Defaults to the full sequence.
- `wt.select(symbol, k)` — 0-based index of the 1-based `k`-th occurrence of `symbol`.
- `wt.range_quantile(ql, qr, k)` — the 0-based `k`-th smallest symbol in `wt[ql:qr]`. Range is half-open and clamped to `[0, len(wt)]`.

## Running the tests

```
PYTHONPATH=src python -m unittest discover -s tests
```

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.


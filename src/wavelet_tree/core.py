"""Core wavelet tree implementation.

A wavelet tree stores a sequence of symbols and answers rank/select queries
in O(alphabet_bits) time using O(n log A) bits. This implementation uses a
binary tree over the alphabet range; each node holds a bit-vector telling
which children each element goes to (left for 0, right for 1).

Design choices
--------------
* The alphabet is the compact integer range [min_symbol, max_symbol]. Only
  symbols actually present influence node construction; unused gaps still
  narrow the tree because we split at the midpoint of [lo, hi].
* Bit-vectors use Python ``int`` as packed bit arrays via the [hi, lo)
  convention: bit ``i`` of node.mask is 1 if symbol ``i`` of the node's
  subsequence routes to the right child. This avoids list-of-bool overhead
  and keeps ``rank`` O(1) per level via int.bit_count on a shifted mask.
* ``select`` walks level by level carrying the 0/1 bit of the queried symbol,
  so it never needs per-node precomputed select structures.
"""

from __future__ import annotations


class _Node:
    """Internal wavelet-tree node.

    Attributes
    ----------
    mask : int
        Bit-vector with ``len`` meaningful bits. Bit ``i`` (counting from the
        LSB of the *position* index, i.e. bit ``len-1-i`` in the int) is 1 if
        the i-th element of this node's subsequence routes right.
    length : int
        Number of elements stored at this node.
    ones : int
        Total number of 1-bits in ``mask``. Cached so rank queries avoid
        recomputing bit_count on every call.
    left, right : Optional[_Node]
        Children for the 0 and 1 routes. Both None at a leaf (single-symbol
        alphabet).
    """

    __slots__ = ("mask", "length", "ones", "left", "right")

    def __init__(self) -> None:
        self.mask: int = 0
        self.length: int = 0
        self.ones: int = 0
        self.left: _Node | None = None
        self.right: _Node | None = None


class WaveletTree:
    """A wavelet tree over an integer alphabet.

    Supports:
    * ``rank(symbol, upto)`` -- count of ``symbol`` in ``self[:upto]``
    * ``select(symbol, k)`` -- index of the k-th (1-based) ``symbol``
    * ``range_quantile(ql, qr, k)`` -- the (0-based) k-th smallest symbol in
      ``self[ql:qr]``

    Symbols may be any integers (negative or non-negative). The tree is built
    once, at construction time, from the full sequence. An empty sequence is
    permitted; all queries on it raise ``IndexError``.
    """

    __slots__ = ("_root", "_lo", "_hi", "_size")

    def __init__(self, data) -> None:
        """Build the tree from an iterable of integers.

        Parameters
        ----------
        data : iterable of int
            The sequence to index. Materialised eagerly into a list; the tree
            retains only packed per-node masks, so after construction the
            original values are not held.
        """
        seq = list(data)
        if any(not isinstance(s, int) or isinstance(s, bool) for s in seq):
            raise TypeError("WaveletTree only supports int symbols (not bool)")
        self._size = len(seq)
        if self._size == 0:
            self._root = None
            self._lo = 0
            self._hi = 0
            return
        lo = min(seq)
        hi = max(seq)
        self._lo = lo
        self._hi = hi
        self._root = self._build(seq, lo, hi)

    # ------------------------------------------------------------------ build
    def _build(self, seq, lo: int, hi: int) -> _Node:
        """Recursively build the subtree for symbols in ``[lo, hi]``."""
        node = _Node()
        node.length = len(seq)
        if lo == hi:
            # Leaf: all symbols identical, nothing to split.
            node.ones = 0
            node.mask = 0
            return node
        # Split at the midpoint so left gets [lo, mid], right gets [mid+1, hi].
        # This keeps the tree balanced in depth over the alphabet range.
        mid = (lo + hi) >> 1
        mask = 0
        left_seq: list[int] = []
        right_seq: list[int] = []
        for i, s in enumerate(seq):
            if s <= mid:
                left_seq.append(s)
            else:
                right_seq.append(s)
                # Bit (length-1-i) = 1: store from MSB side so prefix masks
                # are cheap (rank_up_to = mask >> (length - upto)).
                mask |= 1 << (node.length - 1 - i)
        node.mask = mask
        node.ones = len(right_seq)
        node.left = self._build(left_seq, lo, mid)
        node.right = self._build(right_seq, mid + 1, hi)
        return node

    # --------------------------------------------------------------- rank_1
    @staticmethod
    def _rank1(node: _Node, upto: int) -> int:
        """Number of 1-bits in ``node`` among its first ``upto`` positions."""
        if upto <= 0:
            return 0
        if upto >= node.length:
            return node.ones
        # Drop low (length-upto) bits; the remaining mask holds exactly the
        # first `upto` bits (MSB-first layout).
        return (node.mask >> (node.length - upto)).bit_count()

    @staticmethod
    def _rank0(node: _Node, upto: int) -> int:
        """Number of 0-bits in ``node`` among its first ``upto`` positions."""
        if upto <= 0:
            return 0
        if upto >= node.length:
            return node.length - node.ones
        return upto - WaveletTree._rank1(node, upto)

    # ------------------------------------------------------------------- API
    def __len__(self) -> int:
        return self._size

    def rank(self, symbol: int, upto: int | None = None) -> int:
        """Count occurrences of ``symbol`` in ``self[:upto]``.

        ``upto`` defaults to ``len(self)``. If ``symbol`` lies outside the
        original alphabet range the answer is 0 (not an error).
        """
        if self._root is None:
            return 0
        if upto is None:
            upto = self._size
        if upto < 0:
            upto = 0
        elif upto > self._size:
            upto = self._size
        if symbol < self._lo or symbol > self._hi:
            return 0
        node = self._root
        lo, hi = self._lo, self._hi
        pos = upto
        while node is not None and lo < hi:
            mid = (lo + hi) >> 1
            if symbol <= mid:
                pos = self._rank0(node, pos)
                node = node.left
                hi = mid
            else:
                pos = self._rank1(node, pos)
                node = node.right
                lo = mid + 1
        return pos

    def select(self, symbol: int, k: int) -> int:
        """Return the index of the ``k``-th (1-based) occurrence of ``symbol``.

        Raises ``ValueError`` if ``k`` is not positive or exceeds the count of
        ``symbol`` in the sequence. The returned index is 0-based.
        """
        if k < 1:
            raise ValueError("k is 1-based and must be >= 1")
        if self._root is None:
            raise ValueError("symbol not present: empty sequence")
        if symbol < self._lo or symbol > self._hi:
            raise ValueError("symbol not present: outside alphabet range")
        total = self.rank(symbol)
        if k > total:
            raise ValueError("k exceeds symbol count")
        node = self._root
        lo, hi = self._lo, self._hi
        pos = k  # k-th occurrence within the current subsequence
        path: list[tuple[_Node, int, int, int, bool]] = []
        while node is not None and lo < hi:
            mid = (lo + hi) >> 1
            if symbol <= mid:
                path.append((node, lo, hi, mid, True))
                node = node.left
                hi = mid
            else:
                path.append((node, lo, hi, mid, False))
                node = node.right
                lo = mid + 1
        # Walk back up, converting "k-th occurrence in child" into an index
        # within the parent's subsequence using rank/select inversion.
        for pnode, _plo, _phi, _pmid, went_left in reversed(path):
            if went_left:
                # Find the position whose rank0 == pos. Scan-free: select the
                # pos-th 0-bit of pnode by walking its mask.
                pos = self._select0(pnode, pos)
            else:
                pos = self._select1(pnode, pos)
        return pos - 1  # 1-based inside node -> 0-based in the full sequence

    @staticmethod
    def _select1(node: _Node, k: int) -> int:
        """Position (1-based) of the k-th 1-bit in ``node``."""
        count = 0
        pos = 0
        for i in range(node.length):
            if (node.mask >> (node.length - 1 - i)) & 1:
                count += 1
                if count == k:
                    return i + 1
        raise ValueError("internal: select1 ran off the end")

    @staticmethod
    def _select0(node: _Node, k: int) -> int:
        """Position (1-based) of the k-th 0-bit in ``node``."""
        count = 0
        for i in range(node.length):
            if not ((node.mask >> (node.length - 1 - i)) & 1):
                count += 1
                if count == k:
                    return i + 1
        raise ValueError("internal: select0 ran off the end")

    def range_quantile(self, ql: int, qr: int, k: int) -> int:
        """Return the (0-based) ``k``-th smallest symbol in ``self[ql:qr]``.

        Parameters
        ----------
        ql, qr : int
            Half-open range ``[ql, qr)``. Clamped to ``[0, len(self)]``.
        k : int
            0-based rank of the quantile (``k=0`` is the minimum).

        Raises ``ValueError`` if the range is empty or ``k`` is out of range.
        """
        if self._root is None:
            raise ValueError("empty sequence")
        if ql < 0:
            ql = 0
        if qr > self._size:
            qr = self._size
        if ql >= qr:
            raise ValueError("empty range")
        if k < 0 or k >= qr - ql:
            raise ValueError("k out of range")
        node = self._root
        lo, hi = self._lo, self._hi
        left_q = ql
        right_q = qr
        while lo < hi:
            mid = (lo + hi) >> 1
            # How many elements in [left_q, right_q) route left?
            left_count = self._rank0(node, right_q) - self._rank0(node, left_q)
            if k < left_count:
                left_q = self._rank0(node, left_q)
                right_q = self._rank0(node, right_q)
                node = node.left
                hi = mid
            else:
                k -= left_count
                left_q = self._rank1(node, left_q)
                right_q = self._rank1(node, right_q)
                node = node.right
                lo = mid + 1
        return lo

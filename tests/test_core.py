import unittest

from wavelet_tree import WaveletTree


class TestConstruction(unittest.TestCase):
    def test_empty_sequence(self):
        wt = WaveletTree([])
        self.assertEqual(len(wt), 0)
        self.assertEqual(wt.rank(0), 0)
        self.assertEqual(wt.rank(5, 10), 0)
        with self.assertRaises(ValueError):
            wt.select(0, 1)
        with self.assertRaises(ValueError):
            wt.range_quantile(0, 0, 0)

    def test_rejects_bool_symbols(self):
        # bool is a subclass of int; we explicitly reject it to avoid silent
        # surprises, since True/False carry their own semantics.
        with self.assertRaises(TypeError):
            WaveletTree([1, 2, True])

    def test_accepts_negative_integers(self):
        wt = WaveletTree([-3, -1, -2, -3, 0])
        self.assertEqual(len(wt), 5)
        self.assertEqual(wt.rank(-3), 2)
        self.assertEqual(wt.rank(-3, 3), 1)
        self.assertEqual(wt.rank(0), 1)


class TestRank(unittest.TestCase):
    def test_classic_example(self):
        # Sequence: banana -> b a n a n a  mapped to ints 0..2
        seq = [0, 1, 2, 1, 2, 1]
        wt = WaveletTree(seq)
        self.assertEqual(wt.rank(1, 6), 3)  # count of 1 in whole seq
        self.assertEqual(wt.rank(1, 3), 1)  # only the second element
        self.assertEqual(wt.rank(2, 6), 2)
        self.assertEqual(wt.rank(0, 6), 1)
        self.assertEqual(wt.rank(0, 0), 0)

    def test_rank_outside_alphabet(self):
        wt = WaveletTree([5, 7, 5, 9])
        self.assertEqual(wt.rank(4), 0)  # below min
        self.assertEqual(wt.rank(10), 0)  # above max
        self.assertEqual(wt.rank(6), 0)  # in range but absent

    def test_rank_clamps_upto(self):
        wt = WaveletTree([1, 2, 3])
        self.assertEqual(wt.rank(1, 100), 1)
        self.assertEqual(wt.rank(1, -5), 0)

    def test_rank_single_symbol(self):
        wt = WaveletTree([4, 4, 4, 4])
        self.assertEqual(wt.rank(4, 0), 0)
        self.assertEqual(wt.rank(4, 1), 1)
        self.assertEqual(wt.rank(4, 4), 4)
        self.assertEqual(wt.rank(4, 2), 2)


class TestSelect(unittest.TestCase):
    def test_basic_select(self):
        seq = [0, 1, 2, 1, 2, 1]
        wt = WaveletTree(seq)
        self.assertEqual(wt.select(1, 1), 1)
        self.assertEqual(wt.select(1, 2), 3)
        self.assertEqual(wt.select(1, 3), 5)
        self.assertEqual(wt.select(0, 1), 0)
        self.assertEqual(wt.select(2, 1), 2)
        self.assertEqual(wt.select(2, 2), 4)

    def test_select_invalid_k(self):
        wt = WaveletTree([1, 1, 1])
        with self.assertRaises(ValueError):
            wt.select(1, 0)
        with self.assertRaises(ValueError):
            wt.select(1, -1)
        with self.assertRaises(ValueError):
            wt.select(1, 4)  # only 3 present

    def test_select_symbol_not_present(self):
        wt = WaveletTree([1, 2, 3])
        with self.assertRaises(ValueError):
            wt.select(4, 1)  # outside range
        with self.assertRaises(ValueError):
            wt.select(5, 1)  # in range, absent -> count 0

    def test_select_returns_index_in_full_sequence(self):
        # Multi-level path: alphabet needs at least 3 symbols to exercise
        # the recursive walk-back.
        wt = WaveletTree([2, 0, 1, 0, 2])
        self.assertEqual(wt.select(0, 1), 1)
        self.assertEqual(wt.select(0, 2), 3)
        self.assertEqual(wt.select(1, 1), 2)
        self.assertEqual(wt.select(2, 1), 0)
        self.assertEqual(wt.select(2, 2), 4)


class TestRangeQuantile(unittest.TestCase):
    def test_classic_small(self):
        seq = [3, 1, 4, 1, 5, 9, 2, 6]
        wt = WaveletTree(seq)
        self.assertEqual(wt.range_quantile(0, 8, 0), 1)
        self.assertEqual(wt.range_quantile(0, 8, 7), 9)
        self.assertEqual(wt.range_quantile(2, 5, 0), 1)
        self.assertEqual(wt.range_quantile(2, 5, 1), 4)
        self.assertEqual(wt.range_quantile(2, 5, 2), 5)

    def test_quantile_clamps_range(self):
        wt = WaveletTree([1, 2, 3])
        self.assertEqual(wt.range_quantile(-1, 100, 0), 1)
        self.assertEqual(wt.range_quantile(-1, 100, 2), 3)

    def test_quantile_empty_range(self):
        wt = WaveletTree([1, 2, 3])
        with self.assertRaises(ValueError):
            wt.range_quantile(2, 2, 0)
        with self.assertRaises(ValueError):
            wt.range_quantile(5, 5, 0)

    def test_quantile_k_out_of_range(self):
        wt = WaveletTree([1, 2, 3])
        with self.assertRaises(ValueError):
            wt.range_quantile(0, 3, 3)
        with self.assertRaises(ValueError):
            wt.range_quantile(0, 3, -1)

    def test_quantile_single_element_range(self):
        wt = WaveletTree([5, 7, 5, 9])
        self.assertEqual(wt.range_quantile(0, 1, 0), 5)
        self.assertEqual(wt.range_quantile(1, 2, 0), 7)
        self.assertEqual(wt.range_quantile(2, 3, 0), 5)
        self.assertEqual(wt.range_quantile(3, 4, 0), 9)

    def test_quantile_with_duplicates(self):
        wt = WaveletTree([2, 2, 2, 1, 1])
        self.assertEqual(wt.range_quantile(0, 5, 0), 1)
        self.assertEqual(wt.range_quantile(0, 5, 1), 1)
        self.assertEqual(wt.range_quantile(0, 5, 4), 2)


class TestRoundTrip(unittest.TestCase):
    def test_rank_select_inverse(self):
        seq = [3, 1, 4, 1, 5, 9, 2, 6]
        wt = WaveletTree(seq)
        for s in set(seq):
            total = wt.rank(s)
            for k in range(1, total + 1):
                idx = wt.select(s, k)
                self.assertEqual(seq[idx], s)
                self.assertEqual(wt.rank(s, idx + 1), k)

    def test_quantile_matches_sorted_slice(self):
        seq = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        wt = WaveletTree(seq)
        for ql in range(len(seq)):
            for qr in range(ql + 1, len(seq) + 1):
                sub = sorted(seq[ql:qr])
                for k in range(qr - ql):
                    self.assertEqual(wt.range_quantile(ql, qr, k), sub[k])


if __name__ == "__main__":
    unittest.main()

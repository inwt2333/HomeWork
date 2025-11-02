import heapq
import random
import time
import unittest

from test_utils import load_problem_module, env_flag


class TestKthLargestElement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_problem_module("problem/2_1_Kth_Largest_Element.py")
        cls.Solution = getattr(cls.mod, "Solution", None)

    def _call(self, nums, k):
        sol = self.Solution()
        ans = sol.findKthLargest(nums, k)
        if ans is None:
            self.skipTest("findKthLargest not implemented yet")
        return ans

    def test_examples(self):
        self.assertEqual(self._call([3,2,1,5,6,4], 2), 5)
        self.assertEqual(self._call([3,2,3,1,2,4,5,5,6], 4), 4)

    def test_with_duplicates(self):
        self.assertEqual(self._call([1,1,1,1], 2), 1)

    def test_k_bounds(self):
        nums = [9, 3, 5, 7]
        self.assertEqual(self._call(nums, 1), 9)  # largest
        self.assertEqual(self._call(nums, len(nums)), 3)  # smallest

    @unittest.skipUnless(env_flag("RUN_PERF"), "Set RUN_PERF=1 to run performance tests")
    def test_performance_large(self):
        n = 100000
        random.seed(2025)
        nums = [random.randint(-10000, 10000) for _ in range(n)]
        k = 5000
        start = time.perf_counter()
        ans = self._call(nums[:], k)
        dur = time.perf_counter() - start
        expected = heapq.nlargest(k, nums)[-1]
        self.assertEqual(ans, expected)
        print(f"KthLargest perf: n={n}, k={k}, time={dur:.3f}s")


if __name__ == "__main__":
    unittest.main()

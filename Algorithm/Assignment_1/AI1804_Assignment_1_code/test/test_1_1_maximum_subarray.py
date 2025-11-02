import random
import time
import unittest

from test_utils import load_problem_module, env_flag


class TestMaximumSubarray(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_problem_module("problem/1_1_Maximum_Subarray.py")
        cls.Solution = getattr(cls.mod, "Solution", None)

    def _call(self, nums):
        sol = self.Solution()
        result = sol.maxSubArray(nums)
        if result is None:
            self.skipTest("maxSubArray not implemented yet")
        return result

    def kadane(self, nums):
        best = cur = nums[0]
        for x in nums[1:]:
            cur = max(x, cur + x)
            best = max(best, cur)
        return best

    def test_examples(self):
        nums = [-2,1,-3,4,-1,2,1,-5,4]
        self.assertEqual(self._call(nums), 6)
        self.assertEqual(self._call([1]), 1)
        self.assertEqual(self._call([5,4,-1,7,8]), 23)

    def test_all_negative(self):
        nums = [-8, -3, -6, -2, -5, -4]
        self.assertEqual(self._call(nums), -2)

    def test_all_positive(self):
        nums = [1, 2, 3, 4]
        self.assertEqual(self._call(nums), sum(nums))

    def test_mixed_short(self):
        nums = [2, -1, 2, 3, -9, 5, -2, 1]
        self.assertEqual(self._call(nums), self.kadane(nums))

    @unittest.skipUnless(env_flag("RUN_PERF"), "Set RUN_PERF=1 to run performance tests")
    def test_performance_large_random(self):
        # n up to 1e5, values in [-1e4,1e4]
        n = 100000
        random.seed(42)
        nums = [random.randint(-10000, 10000) for _ in range(n)]
        start = time.perf_counter()
        got = self._call(nums)
        dur = time.perf_counter() - start
        # Validate correctness via Kadane (O(n))
        expected = self.kadane(nums)
        self.assertEqual(got, expected)
        # Soft perf hint (no strict assert to avoid flakiness)
        print(f"MaximumSubarray perf: n={n}, time={dur:.3f}s")


if __name__ == "__main__":
    unittest.main()

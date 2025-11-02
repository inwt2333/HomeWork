import random
import time
import unittest

from test_utils import load_problem_module, env_flag


class TestFindPeakElement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_problem_module("problem/1_2_Find_Peak_Element.py")
        cls.Solution = getattr(cls.mod, "Solution", None)

    def _call(self, nums):
        sol = self.Solution()
        idx = sol.findPeakElement(nums)
        if idx is None:
            self.skipTest("findPeakElement not implemented yet")
        return idx

    def is_peak(self, nums, i):
        n = len(nums)
        left = nums[i-1] if i-1 >= 0 else float('-inf')
        right = nums[i+1] if i+1 < n else float('-inf')
        return nums[i] > left and nums[i] > right

    def test_examples(self):
        self.assertEqual(self._call([1,2,3,1]), 2)
        idx = self._call([1,2,1,3,5,6,4])
        self.assertTrue(idx in (1, 5))

    def test_single_element(self):
        self.assertEqual(self._call([7]), 0)

    def test_monotonic(self):
        # Increasing -> peak at end
        idx = self._call([1,2,3,4,5])
        self.assertEqual(idx, 4)
        # Decreasing -> peak at start
        idx = self._call([5,4,3,2,1])
        self.assertEqual(idx, 0)

    def test_property_check(self):
        nums = [1,3,2,4,1,0,5,2]
        idx = self._call(nums)
        self.assertTrue(self.is_peak(nums, idx))

    @unittest.skipUnless(env_flag("RUN_PERF"), "Set RUN_PERF=1 to run performance tests")
    def test_performance_large_random(self):
        n = 100000
        random.seed(123)
        # Create array with no equal adjacent values
        nums = []
        prev = None
        for _ in range(n):
            x = random.randint(-10**9, 10**9)
            while prev is not None and x == prev:
                x = random.randint(-10**9, 10**9)
            nums.append(x)
            prev = x
        start = time.perf_counter()
        idx = self._call(nums)
        dur = time.perf_counter() - start
        self.assertTrue(self.is_peak(nums, idx))
        print(f"FindPeakElement perf: n={n}, time={dur:.3f}s")


if __name__ == "__main__":
    unittest.main()

import random
import time
import unittest

from test_utils import load_problem_module, env_flag


class TestMajorityElement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_problem_module("problem/1_3_Majority_Element.py")
        cls.Solution = getattr(cls.mod, "Solution", None)

    def _call(self, nums):
        sol = self.Solution()
        ans = sol.majorityElement(nums)
        if ans is None:
            self.skipTest("majorityElement not implemented yet")
        return ans

    def test_examples(self):
        self.assertEqual(self._call([3,2,3]), 3)
        self.assertEqual(self._call([2,2,1,1,1,2,2]), 2)

    def test_negative_numbers(self):
        self.assertEqual(self._call([-1,-1,-1,2,3]), -1)

    def test_majority_at_end(self):
        self.assertEqual(self._call([1,2,3,4,4,4,4]), 4)

    @unittest.skipUnless(env_flag("RUN_PERF"), "Set RUN_PERF=1 to run performance tests")
    def test_performance_large(self):
        n = 100000
        random.seed(7)
        majority = 42
        others = [random.randint(-10**9, 10**9) for _ in range(n//2 - 1)]
        nums = others + [majority] * (n - len(others))
        random.shuffle(nums)
        start = time.perf_counter()
        ans = self._call(nums)
        dur = time.perf_counter() - start
        self.assertEqual(ans, majority)
        print(f"MajorityElement perf: n={n}, time={dur:.3f}s")


if __name__ == "__main__":
    unittest.main()

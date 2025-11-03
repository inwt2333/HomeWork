import random
import time
import unittest

from test_utils import load_problem_module, env_flag


class TestFindDuplicateNumber(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_problem_module("problem/1_4_Find_the_Duplicate_Number.py")
        cls.Solution = getattr(cls.mod, "Solution", None)

    def _call(self, nums):
        sol = self.Solution()
        ans = sol.findDuplicate(nums)
        if ans is None:
            self.skipTest("findDuplicate not implemented yet")
        return ans

    def test_examples(self):
        self.assertEqual(self._call([1,3,4,2,2]), 2)
        self.assertEqual(self._call([3,1,3,4,2]), 3)
        self.assertEqual(self._call([3,3,3,3,3]), 3)

    def test_duplicate_at_edges(self):
        self.assertEqual(self._call([1,1,2,3,4]), 1)
        self.assertEqual(self._call([1,2,3,4,4]), 4)

    @unittest.skipUnless(env_flag("RUN_PERF"), "Set RUN_PERF=1 to run performance tests")
    def test_performance_large(self):
        n = 100000
        random.seed(99)
        dup = 777
        nums = list(range(1, n+1))
        # Insert duplicate by replacing a random position (not the one equal to dup) with dup
        i = random.randrange(n)
        if nums[i] == dup:
            i = (i + 1) % n
        nums[i] = dup
        random.shuffle(nums)
        start = time.perf_counter()
        ans = self._call(nums)
        dur = time.perf_counter() - start
        self.assertEqual(ans, dup)
        print(f"FindDuplicate perf: n={n}, time={dur:.3f}s")


if __name__ == "__main__":
    unittest.main()

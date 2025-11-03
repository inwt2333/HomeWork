import random
import time
import unittest

from test_utils import load_problem_module, env_flag


class TestHospitalClinicAllocationSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_problem_module("problem/3_1_Hospital_Clinic_Allocation_System.py")
        cls.HospitalSystem = getattr(cls.mod, "HospitalSystem", None)
        cls.Solution = getattr(cls.mod, "Solution", None)

    def _assign(self, arrivals, k):
        hs = self.HospitalSystem()
        res = hs.assign_patients_to_clinics(arrivals, k)
        if res is None:
            self.skipTest("assign_patients_to_clinics not implemented yet")
        return res

    def _merge(self, queues):
        hs = self.HospitalSystem()
        res = hs.merge_clinic_queues(queues)
        if res is None:
            self.skipTest("merge_clinic_queues not implemented yet")
        return res

    def _process(self, arrivals, k):
        hs = self.HospitalSystem()
        res = hs.process_hospital_queue(arrivals, k)
        if res is None:
            self.skipTest("process_hospital_queue not implemented yet")
        return res

    def _balance_ok(self, queues):
        sizes = [len(q) for q in queues]
        return (max(sizes) - min(sizes)) <= 1 if sizes else True

    def _clinic_sorted(self, queues):
        return all(all(q[i] <= q[i+1] for i in range(len(q)-1)) for q in queues)

    def test_examples(self):
        arrivals = [4,1,3,2]
        k = 2
        queues = self._assign(arrivals, k)
        self.assertTrue(self._balance_ok(queues))
        self.assertTrue(self._clinic_sorted(queues))
        merged = self._merge(queues)
        self.assertEqual(merged, [1,2,3,4])

        arrivals = [7,1,5,3,2,6,4]
        k = 3
        queues = self._assign(arrivals, k)
        self.assertTrue(self._balance_ok(queues))
        self.assertTrue(self._clinic_sorted(queues))
        merged = self._merge(queues)
        self.assertEqual(merged, [1,2,3,4,5,6,7])

    def test_edge_cases(self):
        # k = 1 (no distribution)
        arrivals = [3,1,2]
        k = 1
        queues = self._assign(arrivals, k)
        self.assertEqual(len(queues), 1)
        self.assertTrue(self._clinic_sorted(queues))
        merged = self._merge(queues)
        self.assertEqual(merged, sorted(arrivals))

        # k > n (some clinics empty)
        arrivals = [2,1]
        k = 5
        queues = self._assign(arrivals, k)
        self.assertEqual(sum(len(q) for q in queues), len(arrivals))
        self.assertTrue(self._balance_ok(queues))
        self.assertTrue(self._clinic_sorted(queues))
        merged = self._merge(queues)
        self.assertEqual(merged, [1,2])

    def test_process_pipeline(self):
        arrivals = [5,2,4,1,3]
        k = 2
        out = self._process(arrivals, k)
        self.assertEqual(out, [1,2,3,4,5])

        # Optional: if Solution wrapper exists, test it too (skip if not implemented)
        if self.Solution is not None:
            sol = self.Solution()
            out2 = sol.process_hospital_queue(arrivals, k)
            if out2 is None:
                self.skipTest("Solution.process_hospital_queue not implemented yet")
            else:
                self.assertEqual(out2, [1,2,3,4,5])

    @unittest.skipUnless(env_flag("RUN_PERF"), "Set RUN_PERF=1 to run performance tests")
    def test_performance_large(self):
        n = 100000
        k = 1000
        arrivals = list(range(1, n+1))
        random.seed(2025)
        random.shuffle(arrivals)
        start = time.perf_counter()
        queues = self._assign(arrivals, k)
        mid = time.perf_counter()
        merged = self._merge(queues)
        end = time.perf_counter()
        self.assertEqual(merged, sorted(arrivals))
        print(
            f"HospitalSystem perf: n={n}, k={k}, assign={mid-start:.3f}s, merge={end-mid:.3f}s, total={end-start:.3f}s"
        )


if __name__ == "__main__":
    unittest.main()

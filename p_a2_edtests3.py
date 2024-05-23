#Pass
import unittest
from assignment2 import allocate

class TestAllocate(unittest.TestCase):
    def test_no_allocation(self):
        preferences = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        officers_per_org = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
        min_shifts = 0
        max_shifts = 30
        self.assertIsNone(allocate(preferences, officers_per_org, min_shifts, max_shifts))

    def test_valid_allocation(self):
        preferences = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        officers_per_org = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        min_shifts = 0
        max_shifts = 30
        allocation = allocate(preferences, officers_per_org, min_shifts, max_shifts)
        self.assertIsNotNone(allocation)
        self.assertEqual(len(allocation), len(preferences))
        self.assertEqual(len(allocation[0]), len(officers_per_org))
        self.assertEqual(len(allocation[0][0]), 30)
        self.assertEqual(len(allocation[0][0][0]), 3)







if __name__ == '__main__':
    unittest.main()
"""
Problem 1-2: Find Peak Element
EN: Return the index of any peak element (strictly greater than neighbors) in O(log n) time.
CN: 在 O(log n) 时间内返回任意一个峰值元素的下标（严格大于相邻元素）。
"""

class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def find_peak_helper(arr, left, right):
            if left == right:
                return left
            mid = int((left + right) / 2)
            if arr[mid] < arr[mid + 1]:
                return find_peak_helper(arr, mid + 1, right)
            else:
                return find_peak_helper(arr, left, mid) 

        if not nums:
            return -1
        return find_peak_helper(nums, 0, len(nums) - 1)

